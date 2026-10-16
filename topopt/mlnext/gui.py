"""Bilingual Streamlit interface for running and studying scientific ML experiments.

Start from repository root:
    python -m streamlit run topopt/mlnext/gui.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from topopt.mlnext.gui_backend import (
    EXPERIMENTS, ROOT, RUNS, TRAINABLE, GENERATIVE,
    active_runs, checkpoints, comparison_task, dataset_info, dataset_task,
    datasets, diffusion_eval_task, diffusion_task, fem_eval_task, fem_task,
    get_run, list_runs, make_study, new_run, read_json, request_cancel,
    summarize_run_files, tail_log, train_task,
)

st.set_page_config(
    page_title="Topology Lab · Scientific ML",
    page_icon="🧬", layout="wide", initial_sidebar_state="expanded",
)


EN = {
    "app": "Topology Lab", "subtitle": "Topology optimization · FEM · Scientific ML",
    "language": "Language", "nav": "Workspace", "overview": "Overview",
    "dataset": "Datasets", "train": "Training", "study": "Comparative study",
    "generative": "Generative design", "fem": "Neural FEM",
    "analysis": "Results & analysis", "runs": "Runs & logs",
    "overview_desc": "A research workbench for physically checked topology optimization and machine learning.",
    "repo": "Project", "local": "Local execution", "run_count": "Runs recorded",
    "dataset_count": "Datasets", "weights_count": "Trained checkpoints",
    "active": "Active job", "idle": "No active jobs", "history": "Recent jobs",
    "no_data": "No generated dataset yet. Start in Datasets.",
    "no_weights": "No checkpoints yet. Train a model first.",
    "dataset_desc": "Generate reference FEM/OC solutions and separated train, validation and out-of-distribution test cases.",
    "dataset_name": "Dataset label", "samples": "Number of FEM cases",
    "nx": "Elements along X", "ny": "Elements along Y",
    "seed": "Random seed", "max_iter": "Maximum OC iterations",
    "generate": "Generate FEM dataset", "dataset_help": "Actual Q4 FEM solves are used. Larger meshes and datasets can take substantial time.",
    "selected_dataset": "Dataset", "manifest": "Dataset manifest",
    "usable": "Converged cases", "requested": "Requested cases",
    "train_desc": "Train a warm-start predictor from scratch using an existing FEM dataset.",
    "model": "Architecture", "epochs": "Maximum epochs",
    "lr": "Learning rate", "device": "Compute device",
    "patience": "Early-stopping patience", "batch": "Batch size",
    "physics": "Differentiable compliance loss weight", "physics_warn":
        "Physics loss uses dense differentiable FEM and is limited to 512 elements.",
    "start_train": "Start training", "study_desc":
        "Train selected architectures on the same dataset and compare against OC and Ridge on held-out cases.",
    "models": "Compare architectures", "seeds": "Random initialization seeds",
    "study_name": "Study label", "run_study": "Start comparative study",
    "benchmark": "Evaluate existing checkpoints", "choose_checkpoints":
        "Select saved checkpoints", "launch_benchmark": "Run OC / Ridge / neural benchmark",
    "gen_desc": "Conditional diffusion models produce candidate densities; FEM ranks candidates and OC refines them.",
    "diff_arch": "Diffusion architecture", "steps": "Diffusion steps",
    "start_diff": "Train generative model", "eval_diff": "Evaluate diffusion samples",
    "samples_per_case": "Candidates per test case", "evaluate": "Evaluate",
    "fem_desc": "Train nodal displacement predictions to initialize iterative Jacobi-preconditioned CG; all solutions still require equilibrium checks.",
    "train_fem": "Train Neural FEM", "test_fem": "Evaluate CG acceleration",
    "analytics_desc": "Browse historical and new experiments. Test-set compliance, FEM calls, training curves and numerical quality remain separate.",
    "source": "Results source", "historical": "Historical Ridge/U-Net results",
    "artifact": "Artifact", "metric": "Metric", "missing": "No compatible result file found.",
    "summary": "Summary", "methods": "Method comparison", "learning": "Learning curves",
    "metrics": "Numerical metrics", "records": "Per-case records",
    "download_csv": "Download CSV", "download_json": "Download JSON",
    "runs_desc": "Processes run locally, independently of the browser tab. Opening this page does not restart work.",
    "select_run": "Select run", "status": "Status", "created": "Created",
    "step": "Current step", "progress": "Completed steps", "log": "Live process output",
    "refresh": "Refresh status", "cancel": "Request cancellation",
    "cancel_note": "Cancellation is cooperative; the running subprocess is terminated by its local worker.",
    "files": "Produced files", "error": "Error", "no_runs": "No runs have been started.",
    "button_running": "Another experiment is active. Finish or cancel it before starting a new one.",
    "queued": "queued", "running": "running", "completed": "completed",
    "failed": "failed", "cancelled": "cancelled", "interrupted": "interrupted",
    "cancel_requested": "cancel requested", "submitted": "Job started",
    "config_error": "Cannot start job", "open_logs": "Open Runs & logs to follow progress.",
    "training_info": "Training checkpoints are selected on validation only; test results must not be used for tuning.",
    "reference": "Reference FEM/OC baseline",
    "quality": "Quality-gate passes", "fallbacks": "Fallbacks",
    "mean_solves": "Mean FEM solves", "speed": "Mean online time (s)",
    "limitations": "Research limitations", "limits_text":
        "No production-grade solver qualification or experimental validation. Speedups require paired end-to-end measurement; learning-loss improvements do not establish engineering gains.",
    "hint": "Tip", "start_hint": "Suggested workflow: generate dataset → train architectures → run paired comparison → inspect reports.",
    "not_found": "Selected file is unavailable.",
    "tag": "A-Z, 0-9, underscore and hyphen only (max 36 characters).",
    "execution_note": "All runs use local Python and CPU by default. This interface is intended for trusted local use only.",
    "file_preview": "Preview", "raw_report": "Report JSON", "run_type": "Experiment",
    "train_loss": "Training loss", "validation_loss": "Validation loss",
    "training_warning": "Test-set performance must be evaluated only after model selection.",
    "latest": "Latest results", "checkpoints": "Checkpoints",
    "file_view": "File browser", "copy_command": "Launch command",
    "stage": "Stage", "study_limit": "Study may run many training jobs sequentially.",
    "legacy_notice": "Historical figures are from the earlier, narrow cantilever benchmark, not this new OOD dataset.",
    "run_overview": "Recent experiment history", "started": "Started",
    "load_view": "Field view", "case_index": "Case index",
    "field_name": "Field", "sample_split": "Data partition",
    "validation": "validation", "test": "test",
    "trained": "Trained models", "inference": "Evaluation & comparison",
    "fem_solves_label": "FEM solves", "online_seconds_label": "Online seconds",
    "quality_label": "Quality gates", "fallbacks_label": "Fallbacks",
    "early_field": "After five OC iterations", "optimized_field": "Optimized design",
    "preview_too_big": "Preview unavailable for compressed datasets over 256 MiB.",
    "cold_cg": "Conventional CG iterations", "warm_cg": "Neural-start CG iterations",
    "faster_count": "Cases faster including inference",
}
RU = {
    "app": "Topology Lab", "subtitle": "Топологическая оптимизация · FEM · Научное ML",
    "language": "Язык", "nav": "Рабочая область", "overview": "Обзор",
    "dataset": "Датасеты", "train": "Обучение", "study": "Сравнение моделей",
    "generative": "Генеративный дизайн", "fem": "Нейросетевой FEM",
    "analysis": "Результаты и анализ", "runs": "Запуски и журналы",
    "overview_desc": "Исследовательская среда для топологической оптимизации и ML с физической проверкой результатов.",
    "repo": "Проект", "local": "Локальные расчёты", "run_count": "Всего запусков",
    "dataset_count": "Датасетов", "weights_count": "Обученных моделей",
    "active": "Активная задача", "idle": "Нет активных задач", "history": "Последние запуски",
    "no_data": "Датасетов пока нет. Создай первый в разделе «Датасеты».",
    "no_weights": "Сохранённых моделей пока нет. Сначала выполни обучение.",
    "dataset_desc": "Генерация эталонных FEM/OC-расчётов с разделением на обучение, валидацию и независимые OOD-тесты.",
    "dataset_name": "Название датасета", "samples": "Количество FEM-задач",
    "nx": "Элементов по X", "ny": "Элементов по Y",
    "seed": "Начальный seed", "max_iter": "Максимум итераций OC",
    "generate": "Создать FEM-датасет", "dataset_help":
        "Генерация выполняет реальные Q4 FEM-расчёты. Большие сетки и выборки потребуют значительного времени.",
    "selected_dataset": "Датасет", "manifest": "Паспорт датасета",
    "usable": "Сошедшихся задач", "requested": "Запрошено задач",
    "train_desc": "Обучение нейросетевой модели warm start на выбранном FEM-датасете.",
    "model": "Архитектура", "epochs": "Максимальное число эпох",
    "lr": "Скорость обучения", "device": "Вычислительное устройство",
    "patience": "Ожидание до early stopping", "batch": "Размер батча",
    "physics": "Вес дифференцируемого compliance loss", "physics_warn":
        "Физический loss использует плотный дифференцируемый FEM и ограничен сетками до 512 элементов.",
    "start_train": "Запустить обучение", "study_desc":
        "Обучение выбранных архитектур на одинаковом датасете и сравнение с OC и Ridge на независимых тестах.",
    "models": "Сравниваемые архитектуры", "seeds": "Seed инициализации моделей",
    "study_name": "Название исследования", "run_study": "Начать исследование",
    "benchmark": "Проверка существующих моделей", "choose_checkpoints":
        "Выбери сохранённые checkpoints", "launch_benchmark": "Сравнить OC / Ridge / нейросети",
    "gen_desc": "Диффузионные модели генерируют варианты плотности; FEM оценивает их, OC уточняет лучшие.",
    "diff_arch": "Генеративная архитектура", "steps": "Шаги диффузии",
    "start_diff": "Обучить генеративную модель", "eval_diff": "Оценка сгенерированных конструкций",
    "samples_per_case": "Кандидатов на тестовую задачу", "evaluate": "Оценить",
    "fem_desc": "Нейросеть предсказывает узловые перемещения для старта итерационного CG с Jacobi-предобуславливанием; равновесие обязательно проверяется.",
    "train_fem": "Обучить Neural FEM", "test_fem": "Измерить ускорение CG",
    "analytics_desc": "Просмотр прежних и новых экспериментов: compliance, FEM-вызовы, обучение и численные проверки анализируются раздельно.",
    "source": "Источник результатов", "historical": "Исторические результаты Ridge/U-Net",
    "artifact": "Артефакт", "metric": "Метрика", "missing": "Подходящий файл результатов не найден.",
    "summary": "Сводка", "methods": "Сравнение методов", "learning": "Кривые обучения",
    "metrics": "Численные метрики", "records": "Результаты по задачам",
    "download_csv": "Скачать CSV", "download_json": "Скачать JSON",
    "runs_desc": "Процессы выполняются локально, независимо от вкладки браузера. Просмотр страницы не перезапускает расчёты.",
    "select_run": "Выбери запуск", "status": "Статус", "created": "Создан",
    "step": "Текущий этап", "progress": "Завершено этапов",
    "log": "Журнал процесса", "refresh": "Обновить статус", "cancel": "Запросить отмену",
    "cancel_note": "Отмена выполняется отдельным процессом-исполнителем, завершающим расчётную программу.",
    "files": "Созданные файлы", "error": "Ошибка", "no_runs": "Запусков пока нет.",
    "button_running": "Другая задача выполняется. Дождись окончания или отмени её.",
    "queued": "в очереди", "running": "выполняется", "completed": "завершено",
    "failed": "ошибка", "cancelled": "отменено", "interrupted": "прервано",
    "cancel_requested": "запрошена отмена", "submitted": "Задача запущена",
    "config_error": "Не удалось начать задачу", "open_logs": "Ход работы доступен в «Запуски и журналы».",
    "training_info": "Checkpoint выбирается только на валидации; результаты тестов нельзя использовать для подбора модели.",
    "reference": "Эталон FEM/OC",
    "quality": "Пройдено проверок", "fallbacks": "Fallback",
    "mean_solves": "Среднее число FEM-вызовов", "speed": "Среднее время онлайн (с)",
    "limitations": "Ограничения исследований", "limits_text":
        "Это не сертифицированный промышленный решатель и не экспериментальная валидация. Ускорение доказывается полным сопоставимым расчётом, а не снижением training loss.",
    "hint": "Подсказка", "start_hint": "Рекомендуемый процесс: сгенерировать датасет → обучить модели → сравнить методы → изучить отчёты.",
    "not_found": "Выбранный файл недоступен.",
    "tag": "Допустимы латинские буквы, цифры, _ и - (до 36 символов).",
    "execution_note": "Расчёты выполняются локально с помощью Python, по умолчанию на CPU. Интерфейс предназначен для доверенного локального использования.",
    "file_preview": "Предпросмотр", "raw_report": "JSON-отчёт", "run_type": "Эксперимент",
    "train_loss": "Ошибка обучения", "validation_loss": "Ошибка валидации",
    "training_warning": "К независимому тесту обращаемся только после выбора модели.",
    "latest": "Последние результаты", "checkpoints": "Checkpoints",
    "file_view": "Просмотр файлов", "copy_command": "Команда запуска",
    "stage": "Этап", "study_limit": "В исследовании могут последовательно выполняться несколько обучений.",
    "legacy_notice": "Исторические результаты относятся к прежнему узкому бенчмарку консольных балок, а не к новому OOD-датасету.",
    "run_overview": "История экспериментов", "started": "Начало",
    "load_view": "Просмотр полей", "case_index": "Номер задачи",
    "field_name": "Поле", "sample_split": "Часть датасета",
    "validation": "валидация", "test": "тест",
    "trained": "Обученные модели", "inference": "Проверка и сравнение",
    "fem_solves_label": "Число FEM-решений", "online_seconds_label": "Время онлайн (с)",
    "quality_label": "Пройдено проверок", "fallbacks_label": "Fallback",
    "early_field": "После пяти итераций OC", "optimized_field": "Оптимизированная конструкция",
    "preview_too_big": "Предпросмотр недоступен для сжатых датасетов более 256 МиБ.",
    "cold_cg": "Итерации обычного CG", "warm_cg": "Итерации CG с нейросетевым стартом",
    "faster_count": "Задач ускорено с учётом inference",
}

# Ensure translations cannot silently fall back to untranslated English.
assert set(EN) == set(RU)
LANGUAGES = {"English": EN, "Русский": RU}
if "lang" not in st.session_state:
    st.session_state.lang = "English"
T = LANGUAGES[st.session_state.lang]


def tr(key):
    return T[key]


def slug(text):
    import re
    text = text.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,36}", text):
        raise ValueError(tr("tag"))
    return text


def alert(exc):
    st.error(f"{tr('config_error')}: {exc}")


def job(kind, build, config):
    try:
        run_id = new_run(kind, build, config)
    except (ValueError, RuntimeError, OSError) as exc:
        alert(exc)
    else:
        st.success(f"{tr('submitted')}: {run_id}")
        st.info(tr("open_logs"))


def choose_dataset(key="dataset_main"):
    items = datasets()
    if not items:
        st.info(tr("no_data"))
        return None
    path = st.selectbox(tr("selected_dataset"), items,
                        key=key, format_func=lambda p: str(p.relative_to(EXPERIMENTS)))
    try:
        info = dataset_info(path)
        if info.get("counts"):
            st.caption(" · ".join(f"{k}: {v}" for k, v in info["counts"].items()))
    except (ValueError, OSError) as exc:
        st.warning(str(exc))
    return path


def checkpoint_choices(architecture, key):
    items = [p for p, name in checkpoints(architecture)]
    if not items:
        return None
    return st.selectbox(tr("checkpoints"), items,
                        key=key, format_func=lambda p: str(p.relative_to(EXPERIMENTS)))


def device_options():
    try:
        import torch
        return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    except (ImportError, OSError):
        return ["cpu"]


def unavailable():
    if active_runs():
        st.warning(tr("button_running"))
        return True
    return False


def heading(title, description):
    st.subheader(tr(title))
    st.caption(tr(description))


def overview():
    heading("overview", "overview_desc")
    current = active_runs()
    all_runs = list_runs()
    a, b, c, d = st.columns(4)
    a.metric(tr("run_count"), len(all_runs))
    b.metric(tr("dataset_count"), len(datasets()))
    c.metric(tr("weights_count"), len(checkpoints()))
    d.metric(tr("active"), len(current))
    st.info(tr("start_hint"))
    if current:
        for r in current:
            st.warning(f"{r['id']} · {tr(r['status'])} · {r.get('current_step', '')}")
    if all_runs:
        st.markdown("#### " + tr("history"))
        st.dataframe(pd.DataFrame([
            {"id": r["id"], tr("run_type"): r["kind"], tr("status"): tr(r["status"]),
             tr("created"): r["created"]}
            for r in all_runs[:12]]), hide_index=True, use_container_width=True)
    st.markdown("#### " + tr("limitations"))
    st.write(tr("limits_text"))


def dataset_page():
    heading("dataset", "dataset_desc")
    if unavailable():
        return
    with st.form("new_dataset"):
        tag = st.text_input(tr("dataset_name"), value="study_01", help=tr("tag"))
        a, b = st.columns(2)
        with a:
            count = st.number_input(tr("samples"), min_value=12, max_value=100000, value=90, step=12)
            nx = st.selectbox(tr("nx"), [8, 16, 32, 64, 96, 128], index=1)
            seed = st.number_input(tr("seed"), min_value=0, max_value=2147483647, value=17)
        with b:
            ny = st.selectbox(tr("ny"), [4, 8, 16, 32, 48, 64], index=1)
            max_iter = st.number_input(tr("max_iter"), min_value=15, max_value=10000,
                                       value=400, step=25)
        st.caption(tr("dataset_help"))
        submit = st.form_submit_button(tr("generate"), type="primary")
    if submit:
        try:
            name = slug(tag)
            job("dataset", lambda p: [("Generate FEM dataset",
                dataset_task(p / name, count=int(count), nx=nx, ny=ny,
                             seed=int(seed), max_iter=int(max_iter)))],
                {"tag": name, "samples": int(count), "nx": nx, "ny": ny,
                 "seed": int(seed), "max_iter": int(max_iter)})
        except ValueError as exc:
            alert(exc)
    if datasets():
        st.divider()
        path = choose_dataset("inspect_ds")
        if path:
            with st.expander(tr("manifest"), expanded=True):
                st.json(dataset_info(path))
            view_density(path)


def view_density(path):
    # NPZ does not support lazy slicing of compressed arrays. Keep browser
    # previews bounded even when the research dataset itself is much larger.
    if Path(path).stat().st_size > 256 * 1024 * 1024:
        st.info(tr("preview_too_big"))
        return
    try:
        # Safe NPZ read (allow_pickle=False); do not deserialize checkpoints.
        with np.load(path, allow_pickle=False) as ar:
            targets = ar["targets"]
            splits = ar["split"].astype(str)
            early = ar["features"][:, 0]
            max_cases = len(targets)
            if max_cases < 1:
                return
            idx = st.slider(tr("case_index"), 0, max_cases - 1, 0)
            field = st.radio(tr("field_name"), ["early_field", "optimized_field"],
                             format_func=tr, horizontal=True)
            density = (early if field == "early_field" else targets)[idx]
            st.caption(f"{tr('sample_split')}: {splits[idx]} · {density.shape[1]} × {density.shape[0]}")
            st.image(np.flipud(np.uint8(np.clip(density, 0, 1) * 255)),
                     clamp=True, width=680)
    except (OSError, ValueError, KeyError) as exc:
        st.warning(str(exc))


def train_page():
    heading("train", "train_desc")
    ds = choose_dataset("train_ds")
    if ds is None or unavailable():
        return
    with st.form("train_neural"):
        arch = st.selectbox(tr("model"), TRAINABLE)
        a, b = st.columns(2)
        with a:
            epochs = st.number_input(tr("epochs"), min_value=1, max_value=20000,
                                     value=400, step=25)
            patience = st.number_input(tr("patience"), min_value=1, max_value=20000,
                                       value=90, step=10)
        with b:
            batch = st.number_input(tr("batch"), min_value=1, max_value=1024,
                                    value=8)
            seed = st.number_input(tr("seed"), min_value=0, max_value=2147483647,
                                   value=17)
        lr = st.number_input(tr("lr"), min_value=0.000001, max_value=1.0,
                             value=0.001, format="%.6f")
        device = st.selectbox(tr("device"), device_options(), key="train_device")
        phys = st.number_input(tr("physics"), min_value=0., max_value=100.,
                               value=0., format="%.4f",
                               help=tr("physics_warn"),
                               disabled=arch != "unet")
        st.caption(tr("training_info"))
        clicked = st.form_submit_button(tr("start_train"), type="primary")
    if clicked:
        job("train", lambda p: [(f"Train {arch}",
            train_task(ds, p / arch, arch, epochs=int(epochs),
                       patience=int(patience), batch_size=int(batch),
                       seed=int(seed), lr=lr, device=device,
                       physics_weight=phys if arch == "unet" else 0))],
            {"dataset": str(ds), "architecture": arch, "epochs": int(epochs),
             "patience": int(patience), "batch_size": int(batch), "seed": int(seed),
             "physics_weight": phys, "learning_rate": lr, "device": device})


def study_page():
    heading("study", "study_desc")
    ds = choose_dataset("study_ds")
    if ds is None or unavailable():
        return
    st.markdown("#### " + tr("study_name"))
    with st.form("multi_study"):
        selected = st.multiselect(tr("models"), TRAINABLE,
                                  default=["unet", "gnn", "fno"])
        raw_seeds = st.text_input(tr("seeds"), value="17, 23, 41")
        a, b, c = st.columns(3)
        epochs = a.number_input(tr("epochs"), value=400, min_value=1, max_value=20000)
        patience = b.number_input(tr("patience"), value=90, min_value=1, max_value=20000)
        batch = c.number_input(tr("batch"), value=8, min_value=1, max_value=1024)
        lr = st.number_input(tr("lr"), min_value=0.000001, max_value=1.0,
                             value=0.001, format="%.6f", key="study_lr")
        device = st.selectbox(tr("device"), device_options(), key="study_device")
        phys = st.number_input(tr("physics"), min_value=0., max_value=100.,
                               value=0., help=tr("physics_warn"))
        st.caption(tr("study_limit"))
        submitted = st.form_submit_button(tr("run_study"), type="primary")
    if submitted:
        try:
            seeds = tuple(int(v.strip()) for v in raw_seeds.split(",") if v.strip())
            if not seeds or any(v < 0 or v > 2147483647 for v in seeds):
                raise ValueError("Invalid random seeds")
            job("study",
                lambda p: make_study(ds, p, selected, seeds=seeds, epochs=int(epochs),
                                     patience=int(patience), batch_size=int(batch),
                                     physics_weight=phys, lr=lr, device=device),
                {"dataset": str(ds), "models": selected, "seeds": seeds,
                 "epochs": int(epochs), "physics_weight": phys,
                 "learning_rate": lr, "device": device})
        except ValueError as exc:
            alert(exc)
    st.divider()
    st.markdown("#### " + tr("benchmark"))
    with st.form("evaluate_models"):
        cp = {}
        for name in TRAINABLE:
            available = [p for p, arch in checkpoints(name)]
            if available:
                labels = ["—"] + [str(p.relative_to(EXPERIMENTS)) for p in available]
                selection = st.selectbox(name, labels, key=f"select_{name}")
                if selection != "—":
                    cp[name] = EXPERIMENTS / selection
        eval_clicked = st.form_submit_button(tr("launch_benchmark"))
    if eval_clicked:
        job("comparison", lambda p: [("OC, Ridge & neural comparison",
            comparison_task(ds, p / "comparison.csv", cp))],
            {"dataset": str(ds), "checkpoints": {k: str(v) for k, v in cp.items()}})


def generative_page():
    heading("generative", "gen_desc")
    ds = choose_dataset("gen_ds")
    if ds is None or unavailable():
        return
    with st.form("gen_train"):
        architecture = st.selectbox(tr("diff_arch"), GENERATIVE)
        a, b = st.columns(2)
        epochs = a.number_input(tr("epochs"), value=300, min_value=1, max_value=20000)
        steps = b.number_input(tr("steps"), value=50, min_value=2, max_value=1000)
        device = st.selectbox(tr("device"), device_options(), key="diff_train_device")
        submitted = st.form_submit_button(tr("start_diff"), type="primary")
    if submitted:
        job("diffusion_train", lambda p: [(f"Train diffusion {architecture}",
            diffusion_task(ds, p / architecture, architecture, int(epochs), int(steps), device))],
            {"dataset": str(ds), "architecture": architecture,
             "epochs": int(epochs), "steps": int(steps), "device": device})
    st.divider()
    st.markdown("#### " + tr("eval_diff"))
    available = [p for p, name in checkpoints() if name in GENERATIVE]
    if not available:
        st.info(tr("no_weights"))
        return
    with st.form("gen_evaluate"):
        cp = st.selectbox(tr("checkpoints"), available,
                          format_func=lambda p: str(p.relative_to(EXPERIMENTS)))
        samples = st.number_input(tr("samples_per_case"), min_value=1,
                                  max_value=128, value=4)
        eval_device = st.selectbox(tr("device"), device_options(), key="diff_eval_device")
        clicked = st.form_submit_button(tr("evaluate"))
    if clicked:
        job("diffusion_eval", lambda p: [("FEM evaluation of generative candidates",
            diffusion_eval_task(ds, cp, p / "diffusion.csv", int(samples), eval_device))],
            {"dataset": str(ds), "checkpoint": str(cp),
             "samples": int(samples), "device": eval_device})


def fem_page():
    heading("fem", "fem_desc")
    ds = choose_dataset("fem_ds")
    if ds is None or unavailable():
        return
    with st.form("train_fem"):
        epochs = st.number_input(tr("epochs"), min_value=1,
                                 max_value=20000, value=300)
        device = st.selectbox(tr("device"), device_options(), key="fem_train_device")
        train_clicked = st.form_submit_button(tr("train_fem"), type="primary")
    if train_clicked:
        job("fem_train", lambda p: [("Train FEM displacement predictor",
            fem_task(ds, p / "fem", int(epochs), device))],
            {"dataset": str(ds), "epochs": int(epochs), "device": device})
    st.divider()
    cp = checkpoint_choices("fem", "fem_checkpoint")
    if cp is None:
        st.info(tr("no_weights"))
        return
    eval_device = st.selectbox(tr("device"), device_options(), key="fem_eval_device")
    if st.button(tr("test_fem"), type="secondary"):
        job("fem_eval", lambda p: [("Evaluate CG and neural warm start",
            fem_eval_task(ds, cp, p / "fem_eval.json", eval_device))],
            {"dataset": str(ds), "checkpoint": str(cp), "device": eval_device})


def summarize_metrics(data):
    """Show only fields actually present in the report; never infer quality."""
    if not isinstance(data, dict):
        return
    models = data.get("model_summary")
    if isinstance(models, dict):
        frame = pd.DataFrame.from_dict(models, orient="index")
        if not frame.empty:
            st.dataframe(frame, use_container_width=True)
            cols = [c for c in ("mean_fe_solves", "mean_seconds")
                    if c in frame and pd.api.types.is_numeric_dtype(frame[c])]
            if cols:
                st.bar_chart(frame[cols])
    if "cases" in data and "mean_cold_iterations" in data:
        left, right = st.columns(2)
        left.metric(tr("cold_cg"), round(data["mean_cold_iterations"], 2))
        right.metric(tr("warm_cg"), round(data["mean_warm_iterations"], 2))
        st.metric(tr("faster_count"), data.get("faster_including_inference", "—"))
        if data.get("results"):
            st.dataframe(pd.DataFrame(data["results"]), hide_index=True)
    if "history" in data and isinstance(data["history"], list):
        history = pd.DataFrame(data["history"])
        if not history.empty:
            st.markdown("#### " + tr("learning"))
            use_cols = [c for c in (
                "train_loss", "val_mse", "training_mse",
                "validation_compliance_ratio", "train_noise_mse",
                "validation_noise_mse", "validation_mse")
                if c in history]
            if use_cols:
                history = history.set_index("epoch") if "epoch" in history else history
                st.line_chart(history[use_cols], use_container_width=True)
            with st.expander(tr("records")):
                st.dataframe(history, hide_index=False, use_container_width=True)


def historical():
    path = ROOT / "evidence_v2" / "families" / "summary.json"
    if not path.is_file():
        st.info(tr("missing"))
        return
    st.info(tr("legacy_notice"))
    values = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame({
        name: {tr("fem_solves_label"): info.get("mean_fe_solves"),
               tr("online_seconds_label"): info.get("mean_seconds"),
               tr("quality_label"): info.get("final_gate_passes"),
               tr("fallbacks_label"): info.get("fallbacks")}
        for name, info in values.items()
        if isinstance(info, dict)
    }).T
    st.dataframe(frame, use_container_width=True)
    if tr("fem_solves_label") in frame:
        st.bar_chart(frame[[tr("fem_solves_label")]])
    with st.expander(tr("raw_report")):
        st.json(values)


def results_page():
    heading("analysis", "analytics_desc")
    choices = [tr("historical"), tr("latest")]
    mode = st.radio(tr("source"), choices, horizontal=True)
    if mode == choices[0]:
        historical()
        return
    report_files = sorted(
        [p for p in EXPERIMENTS.rglob("*")
         if p.is_file() and p.suffix in (".csv", ".json")
         and p.name not in ("run.json", "commands.json", "experiment_contract.json")
         and ".git" not in p.parts],
        key=lambda p: p.stat().st_mtime, reverse=True,
    ) if EXPERIMENTS.is_dir() else []
    if not report_files:
        st.info(tr("missing"))
        return
    path = st.selectbox(tr("artifact"), report_files,
                        format_func=lambda p: str(p.relative_to(EXPERIMENTS)))
    if path.suffix == ".json":
        try:
            data = read_json(path)
            summarize_metrics(data)
            with st.expander(tr("raw_report")):
                st.json(data)
            st.download_button(tr("download_json"), path.read_bytes(),
                               file_name=path.name, mime="application/json")
        except (OSError, ValueError, TypeError) as exc:
            st.error(str(exc))
    else:
        try:
            df = pd.read_csv(path)
            st.dataframe(df, hide_index=True, use_container_width=True)
            if not df.empty and "method" in df and "fe_solves" in df:
                chart = df.groupby("method", as_index=True)["fe_solves"].mean().sort_values()
                st.bar_chart(chart)
            if not df.empty and "case_id" in df and "final_compliance_ratio" in df:
                st.scatter_chart(df, x="case_id", y="final_compliance_ratio",
                                 color="method" if "method" in df else None)
            st.download_button(tr("download_csv"), path.read_bytes(),
                               file_name=path.name, mime="text/csv")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            st.error(str(exc))
    st.markdown("#### " + tr("load_view"))
    dataset_paths = datasets()
    if dataset_paths:
        with st.expander(tr("load_view")):
            ds = choose_dataset("viz_dataset")
            if ds:
                view_density(ds)


@st.fragment(run_every="5s")
def runs_page():
    heading("runs", "runs_desc")
    if st.button("↻ " + tr("refresh")):
        st.rerun()
    all_runs = list_runs()
    if not all_runs:
        st.info(tr("no_runs"))
        return
    selected = st.selectbox(
        tr("select_run"), [r["id"] for r in all_runs],
        format_func=lambda ident: f"{ident} · " +
        next((tr(r["status"]) + " · " + r["kind"] for r in all_runs
              if r["id"] == ident), ""),
    )
    r = get_run(selected)
    a, b, c = st.columns(3)
    a.metric(tr("status"), tr(r["status"]))
    b.metric(tr("progress"), f"{min(r.get('step_index', 0), len(r['steps']))}/{len(r['steps'])}")
    c.metric(tr("step"), r.get("current_step") or "—")
    st.caption(f"{tr('created')}: {r['created']} · {tr('started')}: {r.get('started') or '—'}")
    progress = min(1., max(0., r.get("step_index", 0) / max(1, len(r["steps"]))))
    st.progress(progress)
    if r["status"] in ("queued", "running", "cancel_requested"):
        if st.button(tr("cancel"), type="secondary"):
            if request_cancel(selected):
                st.warning(tr("cancel_note"))
    if r.get("error"):
        st.error(str(r["error"]))
    with st.expander(tr("stage")):
        for step in r.get("steps", []):
            st.code(" ".join(step["argv"]), language="text")
    st.markdown("#### " + tr("log"))
    st.code(tail_log(selected) or "—", language="text")
    files = summarize_run_files(r)
    if files:
        st.markdown("#### " + tr("files"))
        names = [str(p.relative_to(Path(r["directory"]))) for p in files]
        st.dataframe(pd.DataFrame({"file": names}), hide_index=True,
                     use_container_width=True)
        viewable = [p for p in files if p.suffix.lower() in (".csv", ".json", ".png")
                    and p.stat().st_size < 12 * 1024 * 1024]
        if viewable:
            pick = st.selectbox(tr("file_preview"), viewable,
                                format_func=lambda p: str(p.relative_to(Path(r["directory"]))))
            if pick.suffix.lower() == ".csv":
                st.dataframe(pd.read_csv(pick), use_container_width=True)
            elif pick.suffix.lower() == ".json":
                obj = json.loads(pick.read_text(encoding="utf-8"))
                summarize_metrics(obj)
                with st.expander(tr("raw_report")):
                    st.json(obj)
            elif pick.suffix.lower() == ".png":
                st.image(str(pick))


with st.sidebar:
    st.title("🧬 " + tr("app"))
    st.caption(tr("subtitle"))
    selected_lang = st.selectbox(
        tr("language"), list(LANGUAGES), key="language_selector",
        index=list(LANGUAGES).index(st.session_state.lang))
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun()
    st.divider()
    pages = ("overview", "dataset", "train", "study", "generative", "fem", "analysis", "runs")
    current_page = st.radio(tr("nav"), pages, format_func=tr, label_visibility="collapsed")
    st.divider()
    active = active_runs()
    if active:
        for item in active:
            st.warning(f"● {tr(item['status'])}\n\n{item.get('current_step') or item['kind']}")
    else:
        st.caption("● " + tr("idle"))
    st.caption(tr("execution_note"))

st.title(tr("app"))
st.caption(tr("subtitle"))
{
    "overview": overview, "dataset": dataset_page, "train": train_page,
    "study": study_page, "generative": generative_page, "fem": fem_page,
    "analysis": results_page, "runs": runs_page,
}[current_page]()
