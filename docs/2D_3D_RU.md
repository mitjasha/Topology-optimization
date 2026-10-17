# 2D и 3D топологическая оптимизация

## Два независимых численных режима

Проект содержит **два FEM-решателя** с общим алгоритмом OC:

| Возможность | 2D | 3D |
|---|---|---|
| Конечные элементы | Q4 (4 узла, 2 DOF/узел) | Hex8 (8 узлов, 3 DOF/узел) |
| Упругость | Plane stress | Полная трёхмерная линейная упругость |
| Плотности | (ny, nx) | (nz, ny, nx) |
| FEM | Sparse SciPy | Sparse SciPy |
| Оптимизация | SIMP + OC | SIMP + тот же OC |
| ML | Ridge, U-Net, GNN, FNO, diffusion, neural FEM | Ridge, U-Net3D, GNN3D, FNO3D |
| Визуализация | 2D-карта плотности | Объёмные ячейки и поперечный Z-срез |
| Обмен данными | прежние артефакты | legacy VTK CELL_DATA, CSV/NPZ/JSON |
| Диффузионные модели | Реализованы для 2D | Не реализованы для 3D |
| Neural CG | Экспериментальный 2D | Пока не интегрирован для 3D |

**Важно:** изопараметрический Hex8 является новым самостоятельным FEM-компонентом;
ранее полученные 2D результаты Ridge/U-Net сохраняются без изменений.

## Одна инженерная задача с известными параметрами

В GUI открой **«Расчёт одной конструкции»** и выбери 2D или 3D.
Укажи сетку, размеры, положение нагрузки, объём материала,
радиус фильтра и число итераций. Результат включает JSON-отчёт,
историю сходимости, поле плотности и VTK (в 3D).

Рассчитать такой же пример напрямую:

```bash
python -m topopt.mlnext.case_run --dimension 3 --nx 8 --ny 4 --nz 4 --length 2 --width 1 --height 1 --volume 0.5 --load-y 0.5 --load-z 0.5 --radius 0.4 --output artifacts/mlnext/manual3d

python -m topopt.mlnext.case_run --dimension 2 --nx 16 --ny 8 --length 2 --height 1 --volume 0.5 --load-y 0.5 --radius 0.19 --output artifacts/mlnext/manual2d
```

## Первый запуск

Установить зависимости:

```bash
python -m pip install -r requirements-gui.txt
python -m pytest mltests/test_3d.py -v
python -m streamlit run topopt/mlnext/gui.py
```

В боковой панели выбрать **Размерность → 2D / 3D**. В 3D-режиме
можно создать датасет, настроить и провести обучение, запустить
сравнительный эксперимент, осмотреть объёмные топологии.

## Отдельный CLI для 3D

Начальный сеточный уровень 8×4×4:

```bash
python -m topopt.mlnext.data3d --output artifacts/mlnext/solid3d/dataset3d.npz --count 60 --nx 8 --ny 4 --nz 4 --max-iter 400

python -m topopt.mlnext.train3d train \
  --dataset artifacts/mlnext/solid3d/dataset3d.npz \
  --model unet3d --output artifacts/mlnext/solid3d/unet3d \
  --epochs 400 --patience 90

python -m topopt.mlnext.train3d train \
  --dataset artifacts/mlnext/solid3d/dataset3d.npz \
  --model fno3d --output artifacts/mlnext/solid3d/fno3d \
  --epochs 400 --patience 90

python -m topopt.mlnext.train3d train \
  --dataset artifacts/mlnext/solid3d/dataset3d.npz \
  --model gnn3d --output artifacts/mlnext/solid3d/gnn3d \
  --epochs 400 --patience 90

python -m topopt.mlnext.train3d evaluate \
  --dataset artifacts/mlnext/solid3d/dataset3d.npz \
  --output artifacts/mlnext/solid3d/comparison3d.csv \
  --checkpoint unet3d=artifacts/mlnext/solid3d/unet3d/best.pt \
  --checkpoint fno3d=artifacts/mlnext/solid3d/fno3d/best.pt \
  --checkpoint gnn3d=artifacts/mlnext/solid3d/gnn3d/best.pt
```

Для CUDA-доступной видеокарты обучение поддерживает `--device cuda`.
FEM, подготовка датасетов и физическая проверка пока CPU/SciPy.

В Windows используйте те же команды в одной строке либо перенесите
соответствующие параметры в поля GUI.

## 3D-постановка

Решается уравнение равновесия

\[
\nabla\cdot\boldsymbol\sigma + \mathbf f = 0,
\qquad \boldsymbol\sigma=\mathbf D_{3D}\boldsymbol\varepsilon,
\qquad
\boldsymbol\varepsilon=\tfrac12(\nabla\mathbf u + \nabla\mathbf u^\mathsf T).
\]

SIMP:

\[
E_e=E_{\min}+(1-E_{\min})\rho_e^p,\quad
K(\rho)\mathbf u=\mathbf F,\quad
\min_{\rho}\mathbf F^\mathsf T\mathbf u.
\]

Ограничение доли материала накладывается на **физическую плотность**
после фильтра. Элемент Hex8 интегрируется по схеме 2×2×2 точек Гаусса.
Область: параллелепипед, левая грань зафиксирована, на правой приложена
сила по направлению −Z с интерполяцией между соседними узлами.

В данной версии материал имеет условное \(E=1\), толщинного
приближения plane stress нет. Это регулярные сетки, не произвольная CAD
и не экспериментально подтверждённые конструкции.

## Протокол проверки

- Тензор жёсткости Hex8: симметрия, шесть жёстких мод, энергия
  постоянной деформации.
- Глобальный 3D FEM: невязка равновесия, баланс реакций и нагрузок,
  равенство податливости и внутренней энергии.
- Чувствительности SIMP: аналитический градиент относительно плотностей
  и независимые конечные разности.
- OC: сходимость, ограничение объёма, снижение compliance.
- ML: трёхмерные формы входных данных, backprop и соблюдение физического
  quality gate после обязательной FEM-доработки.
- При исследовании более крупных задач: сеточная сходимость,
  зависимость от радиуса фильтра, изменение нагрузок и невиданные
  отношения сторон. Замеры выполняются попарно на одном оборудовании.

**Сроки и показатели ускорения не заявляются без выполненных расчётов.**
Сетка 8×4×4 — проверка workflow, а не достоверная конструкция.

## Ограничения и дальнейшее развитие

1. Проверить сходимость на 8×4×4, 16×8×8 и более мелких элементах;
   исследовать зависимость от физических габаритов и радиуса фильтра.
2. Добавить оболочечные и тетраэдральные сетки, новые виды закрепления,
   точечные/распределённые нагрузки и собственные геометрии.
3. Добавить MMA, мультиматериалы, напряжения, собственные частоты,
   buckling и технологические ограничения.
4. Ускорить крупный Hex8 FEM итерационным решателем, предобуславливанием,
   параллельной сборкой и, при наличии доказанного выигрыша, neural CG.
5. Развивать генеративные diffusion-модели для трёхмерных кандидатов
   после появления качественного 3D-бенчмарка.

3D U-Net, GNN и FNO, реализованные здесь, пока являются архитектурами
исследования, а не подтверждённым улучшением относительно Ridge или OC.
