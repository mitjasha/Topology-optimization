"""Dimension=3 workspace for the bilingual Topology Lab GUI.

The original GUI and numerical code remain the dimension=2 workspace.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from topopt.mlnext import gui_backend as backend


TEXT = {
    "English":{
        "heading":"3D topology optimization","overview":"Overview","dataset":"Datasets",
        "train":"Training","study":"Comparative study","analysis":"Results & analysis",
        "desc":"Real volumetric Hex8 FEM, SIMP optimization and neural warm starts.",
        "overview_desc":"A research environment for three-dimensional solid mechanics.",
        "dataset_desc":"Generate actual 3D FEM/OC references with held-out aspect ratios and load locations.",
        "train_desc":"Train an independent 3D neural network on voxelized FEM fields.",
        "study_desc":"Compare 3D Ridge, U-Net, FNO and GNN through physical FEM refinement.",
        "analysis_desc":"Inspect 3D compliance, convergence, 3D density and trained models.",
        "run_count":"Total runs","dataset_count":"3D datasets","trained":"3D checkpoints",
        "dataset_name":"Dataset label","samples":"Number of 3D FEM cases",
        "nx":"Cells X","ny":"Cells Y","nz":"Cells Z","seed":"Random seed",
        "iter":"Maximum OC iterations","make_dataset":"Generate 3D dataset",
        "select_dataset":"3D dataset","no_data":"No 3D datasets. Generate one first.",
        "model":"Architecture","epochs":"Maximum epochs","patience":"Early-stopping patience",
        "batch":"Batch size","learning_rate":"Learning rate","device":"Device",
        "start_train":"Start 3D training","start_study":"Start comparative 3D study",
        "models":"Architectures","seeds":"Independent seeds (comma separated)",
        "evaluate":"Compare available checkpoints","checkpoints":"Trained models",
        "run_comparison":"Run OC / Ridge / 3D neural benchmark",
        "early":"After five OC iterations","final":"Optimized density",
        "case":"Case","field":"Density field","threshold":"Solid-density threshold",
        "slice":"Z slice","preview":"Volumetric FEM density preview",
        "results":"3D reports","historical":"Previous 2D benchmark is available in 2D mode.",
        "no_reports":"No 3D reports found yet.",
        "quality":"Final FEM quality gates","solves":"FEM calls","learning":"Learning curves",
        "details":"Case metadata","source":"Report source","run":"Execute",
        "files":"Output files","download":"Download file",
        "supported":"Experimental availability",
        "unsupported":"3D diffusion and neural CG are not integrated yet; their existing implementations remain available in 2D mode. 3D U-Net, MeshGNN, FNO, Ridge and OC are supported.",
        "limits":"3D problems require substantially more memory and CPU/GPU time than 2D. Start with 8×4×4 or 12×6×4 cells. This implementation uses linear elastic Hex8 elements and a clamped cantilever; it is not an industrial 3D solver.",
        "busy":"A job is already running. Use Runs & logs.",
        "failed":"Unable to create experiment",
        "created":"Run started",
        "inspect":"Open Runs & logs to view the live progress.",
        "too_big":"Compressed dataset exceeds the interactive preview memory limit.",
    },
    "Русский":{
        "heading":"3D топологическая оптимизация","overview":"Обзор","dataset":"Датасеты",
        "train":"Обучение","study":"Сравнение моделей","analysis":"Результаты и анализ",
        "desc":"Настоящий объёмный FEM Hex8, SIMP-оптимизация и нейросетевой warm start.",
        "overview_desc":"Исследовательская среда для трёхмерной механики твёрдого тела.",
        "dataset_desc":"Генерация 3D FEM/OC-решений с независимыми тестами по геометрии и нагрузкам.",
        "train_desc":"Обучение трёхмерной нейросети на объёмных полях FEM.",
        "study_desc":"Сравнение 3D Ridge, U-Net, FNO и GNN с физической FEM-доработкой.",
        "analysis_desc":"Анализ 3D compliance, сходимости, объёмных плотностей и нейросетей.",
        "run_count":"Всего запусков","dataset_count":"3D-датасетов","trained":"3D-моделей",
        "dataset_name":"Название датасета","samples":"Количество 3D FEM-задач",
        "nx":"Элементов X","ny":"Элементов Y","nz":"Элементов Z","seed":"Начальный seed",
        "iter":"Максимум итераций OC","make_dataset":"Создать 3D-датасет",
        "select_dataset":"3D-датасет","no_data":"3D-датасетов пока нет. Создай первый.",
        "model":"Архитектура","epochs":"Максимум эпох","patience":"Ожидание early stopping",
        "batch":"Размер батча","learning_rate":"Скорость обучения","device":"Устройство",
        "start_train":"Начать обучение 3D","start_study":"Запустить сравнение 3D",
        "models":"Архитектуры","seeds":"Независимые seed (через запятую)",
        "evaluate":"Сравнение обученных моделей","checkpoints":"Обученные модели",
        "run_comparison":"Сравнить OC / Ridge / 3D-нейросети",
        "early":"После пяти итераций OC","final":"Оптимизированная плотность",
        "case":"Задача","field":"Поле плотности","threshold":"Порог твёрдого материала",
        "slice":"Срез Z","preview":"Просмотр объёмной FEM-плотности",
        "results":"3D-отчёты","historical":"Прежний 2D benchmark доступен в режиме 2D.",
        "no_reports":"Отчётов по 3D пока нет.",
        "quality":"Пройдено проверок FEM","solves":"Число FEM-решений","learning":"Кривые обучения",
        "details":"Параметры задачи","source":"Источник отчёта","run":"Запустить",
        "files":"Созданные файлы","download":"Скачать файл",
        "supported":"Доступность технологий",
        "unsupported":"3D diffusion и нейросетевой CG пока не интегрированы. Их существующие версии доступны в 2D. В 3D работают U-Net, MeshGNN, FNO, Ridge и OC.",
        "limits":"3D требует существенно больше памяти и вычислений, чем 2D. Начни с сетки 8×4×4 или 12×6×4. Здесь линейно-упругие Hex8 и консольное закрепление, не промышленный 3D-солвер.",
        "busy":"Другая задача уже выполняется. Открой «Запуски и журналы».",
        "failed":"Не удалось создать эксперимент",
        "created":"Задача запущена",
        "inspect":"Журнал выполнения доступен в «Запуски и журналы».",
        "too_big":"Сжатый датасет слишком большой для интерактивного просмотра.",
    },
}
assert set(TEXT["English"])==set(TEXT["Русский"])


def render(page, language="Русский"):
    t=TEXT[language].__getitem__
    if page not in ("overview","dataset","train","study","analysis"):
        st.info(t("unsupported"))
        return
    st.subheader(t(page))
    st.caption(t(page+"_desc") if page!="overview" else t("overview_desc"))
    st.info(t("limits"))

    def start(kind,builder,config):
        try:
            ident=backend.new_run(kind,builder,config)
            st.success(f"{t('created')}: {ident}")
            st.caption(t("inspect"))
        except (ValueError,RuntimeError,OSError) as exc:
            st.error(f"{t('failed')}: {exc}")

    def choose_dataset(key):
        available=backend.datasets3d()
        if not available:
            st.info(t("no_data"))
            return None
        ds=st.selectbox(t("select_dataset"),available,key=key,
                        format_func=lambda p:str(p.relative_to(backend.EXPERIMENTS)))
        try:
            info=backend.dataset_info(ds)
            st.caption(" · ".join(f"{k}: {v}" for k,v in info.get("counts",{}).items()))
        except (OSError,ValueError) as exc:
            st.warning(str(exc))
        return ds

    if page=="overview":
        left,center,right=st.columns(3)
        left.metric(t("run_count"),len(backend.list_runs()))
        center.metric(t("dataset_count"),len(backend.datasets3d()))
        right.metric(t("trained"),
                     sum(1 for p,name in backend.checkpoints()
                         if name in backend.THREED_ARCHITECTURES))
        st.info(t("supported")+": "+t("unsupported"))
        ds=choose_dataset("overview_3d")
        if ds:
            preview(ds,t)
        return

    if page=="dataset":
        if backend.active_runs():
            st.warning(t("busy"))
        with st.form("generate_3d"):
            label=st.text_input(t("dataset_name"),value="solid3d_01")
            count=st.number_input(t("samples"),min_value=12,max_value=100000,
                                  value=60,step=12)
            a,b,c=st.columns(3)
            nx=a.number_input(t("nx"),min_value=4,max_value=256,value=8,step=2)
            ny=b.number_input(t("ny"),min_value=4,max_value=256,value=4,step=2)
            nz=c.number_input(t("nz"),min_value=4,max_value=256,value=4,step=2)
            seed=st.number_input(t("seed"),min_value=0,max_value=2147483647,value=17)
            max_iter=st.number_input(t("iter"),min_value=15,max_value=10000,value=400)
            submit=st.form_submit_button(t("make_dataset"),type="primary",
                                         disabled=bool(backend.active_runs()))
        if submit:
            import re
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,36}",label):
                st.error("Use 1..36 ASCII letters, digits, '_' or '-'")
            else:
                start("data3d",lambda folder:[("Hex8 FEM dataset",
                      backend.dataset_task3d(folder/label,count=int(count),
                          nx=int(nx),ny=int(ny),nz=int(nz),seed=int(seed),
                          max_iter=int(max_iter)))],
                      {"dimension":3,"mesh":[nx,ny,nz],"count":int(count),"seed":int(seed)})
        if backend.datasets3d():
            st.divider()
            ds=choose_dataset("inspect_3d")
            if ds:
                preview(ds,t)
        return

    if page=="train":
        ds=choose_dataset("train_3d")
        if ds is None:
            return
        with st.form("train_3d"):
            arch=st.selectbox(t("model"),backend.THREED_ARCHITECTURES)
            a,b=st.columns(2)
            epochs=a.number_input(t("epochs"),min_value=1,max_value=20000,value=300)
            patience=b.number_input(t("patience"),min_value=1,max_value=20000,value=80)
            batch=a.number_input(t("batch"),min_value=1,max_value=1024,value=4)
            seed=b.number_input(t("seed"),min_value=0,max_value=2147483647,value=17)
            lr=st.number_input(t("learning_rate"),min_value=.000001,max_value=1.,
                               value=.001,format="%.6f")
            device=st.selectbox(t("device"),device_options())
            clicked=st.form_submit_button(t("start_train"),type="primary",
                                          disabled=bool(backend.active_runs()))
        if clicked:
            start("train3d",lambda folder:[(f"Train {arch}",
                  backend.train_task3d(ds,folder/arch,arch,epochs=int(epochs),
                      patience=int(patience),batch_size=int(batch),
                      seed=int(seed),lr=lr,device=device))],
                  {"dimension":3,"dataset":str(ds),"architecture":arch,
                   "epochs":int(epochs),"device":device})
        return

    if page=="study":
        ds=choose_dataset("study_3d")
        if ds is None:
            return
        with st.form("study_3d"):
            selected=st.multiselect(t("models"),backend.THREED_ARCHITECTURES,
                                    default=list(backend.THREED_ARCHITECTURES))
            seeds=st.text_input(t("seeds"),value="17, 23, 41")
            a,b=st.columns(2)
            epochs=a.number_input(t("epochs"),min_value=1,max_value=20000,value=300)
            patience=b.number_input(t("patience"),min_value=1,max_value=20000,value=80)
            batch=a.number_input(t("batch"),min_value=1,max_value=1024,value=4)
            lr=b.number_input(t("learning_rate"),min_value=.000001,max_value=1.,
                              value=.001,format="%.6f")
            device=st.selectbox(t("device"),device_options())
            clicked=st.form_submit_button(t("start_study"),type="primary",
                                          disabled=bool(backend.active_runs()))
        if clicked:
            try:
                seed_values=tuple(int(v.strip()) for v in seeds.split(",") if v.strip())
                if not seed_values or any(s<0 or s>2147483647 for s in seed_values):
                    raise ValueError("Invalid seeds")
                start("study3d",lambda folder:backend.make_study3d(
                     ds,folder,selected,seeds=seed_values,epochs=int(epochs),
                     patience=int(patience),batch_size=int(batch),lr=lr,device=device),
                     {"dimension":3,"dataset":str(ds),"models":selected,
                      "seeds":seed_values,"epochs":int(epochs),"device":device})
            except ValueError as exc:
                st.error(str(exc))
        st.divider()
        st.markdown("#### "+t("evaluate"))
        checkpoint_paths={}
        with st.form("compare_existing_3d"):
            for name in backend.THREED_ARCHITECTURES:
                available=[p for p,n in backend.checkpoints(name)]
                if available:
                    opts=[None]+available
                    choice=st.selectbox(name,opts,
                                        format_func=lambda p:("—" if p is None
                                            else str(p.relative_to(backend.EXPERIMENTS))),
                                        key=f"checkpoint3d_{name}")
                    if choice is not None:
                        checkpoint_paths[name]=choice
            launch=st.form_submit_button(t("run_comparison"),
                                         disabled=bool(backend.active_runs()))
        if launch:
            start("comparison3d",lambda folder:[("Paired Hex8 OC/Ridge/ML",
                  backend.comparison_task3d(ds,folder/"comparison3d.csv",
                                             checkpoint_paths))],
                  {"dimension":3,"dataset":str(ds),
                   "checkpoints":{k:str(v) for k,v in checkpoint_paths.items()}})
        return

    if page=="analysis":
        choices=sorted(
            [p for p in backend.EXPERIMENTS.rglob("*")
             if p.is_file() and p.suffix in (".csv",".json")
             and ("3d" in str(p.relative_to(backend.EXPERIMENTS)).lower()
                  or "dimension" in p.name)
             and p.name!="run.json"],
            key=lambda p:p.stat().st_mtime,reverse=True) if backend.EXPERIMENTS.exists() else []
        # Training reports live inside run directories and do not always say "3d" in their name.
        for path in backend.EXPERIMENTS.rglob("training.json") if backend.EXPERIMENTS.exists() else ():
            try:
                meta=json.loads(path.read_text(encoding="utf-8"))
                if meta.get("dimension")==3 and path not in choices:
                    choices.append(path)
            except (OSError,ValueError):
                pass
        if choices:
            report=st.selectbox(t("results"),choices,
                                format_func=lambda p:str(p.relative_to(backend.EXPERIMENTS)))
            if report.suffix==".csv":
                frame=pd.read_csv(report)
                st.dataframe(frame,use_container_width=True,hide_index=True)
                if "method" in frame and "fe_solves" in frame:
                    st.bar_chart(frame.groupby("method")["fe_solves"].mean())
            else:
                report_json=json.loads(report.read_text(encoding="utf-8"))
                if "model_summary" in report_json:
                    st.dataframe(pd.DataFrame.from_dict(report_json["model_summary"],
                                 orient="index"),use_container_width=True)
                if "history" in report_json:
                    history=pd.DataFrame(report_json["history"])
                    if not history.empty:
                        st.line_chart(history.set_index("epoch")[
                            [c for c in ("train_loss","val_mse") if c in history]])
                st.json(report_json,expanded=False)
            st.download_button(t("download"),report.read_bytes(),file_name=report.name)
        else:
            st.info(t("no_reports"))
        ds=choose_dataset("analyze3d")
        if ds:
            preview(ds,t)


def device_options():
    try:
        import torch
        if torch.cuda.is_available():
            return ("cpu","cuda")
    except (OSError,ImportError):
        pass
    return ("cpu",)


def preview(dataset,t):
    if Path(dataset).stat().st_size > 256*1024*1024:
        st.info(t("too_big"))
        return
    st.markdown("#### "+t("preview"))
    with np.load(dataset,allow_pickle=False) as data:
        targets=data["targets"]
        early=data["features"][:,0]
        records=data["cases"]
        count=len(targets)
        index=st.slider(t("case"),min_value=0,max_value=count-1,value=0,
                        key="3d_case_index")
        label=st.radio(t("field"),("early","final"),
                       format_func=lambda v:t(v),horizontal=True)
        rho=(early if label=="early" else targets)[index]
        case=json.loads(str(records[index]))["case"]
    threshold=st.slider(t("threshold"),.05,.95,.45,.05,key="3d_threshold")
    nz,ny,nx=rho.shape
    zz,yy,xx=np.mgrid[:nz,:ny,:nx]
    selection=rho>=threshold
    centers=np.column_stack((xx[selection],yy[selection],zz[selection]))
    values=rho[selection]
    if len(centers):
        stride=max(1,int(np.ceil(len(centers)/14000)))
        points=centers[::stride]
        densities=values[::stride]
        fig=go.Figure(go.Scatter3d(
            x=(points[:,0]+.5)*case["length"]/nx,
            y=(points[:,1]+.5)*case["width"]/ny,
            z=(points[:,2]+.5)*case["height"]/nz,
            mode="markers",
            marker=dict(size=max(2,min(10,100/max(nx,ny,nz))),
                        color=densities,colorscale="Viridis",
                        cmin=0,cmax=1,opacity=.85,
                        colorbar=dict(title="Density"))))
        fig.update_layout(margin=dict(l=0,r=0,b=0,t=0),
            scene=dict(xaxis_title="X",yaxis_title="Y",zaxis_title="Z",
                       aspectmode="data"),height=490)
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.warning("No cells exceed the current material threshold.")
    zslice=st.slider(t("slice"),0,nz-1,nz//2,key="3d_slice")
    st.image(np.flipud(np.uint8(rho[zslice]*255)),width=680,clamp=True)
    with st.expander(t("details")):
        st.json(case)
