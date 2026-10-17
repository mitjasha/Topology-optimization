"""Bilingual 2D/3D single-case workspace with directly controlled FEM conditions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from topopt.mlnext import gui_backend as backend


TEXT={
 "English":{
  "title":"Single-case solver","description":"Run an actual FEM/OC topology optimization with chosen geometry, mesh, material volume and load position.",
  "nx":"Elements along X","ny":"Elements along Y","nz":"Elements along Z",
  "length":"Length X","width":"Width Y","height":"Height Z / plate height",
  "volume":"Material volume fraction","load_y":"Load Y (normalized)",
  "load_z":"Load Z (normalized)","radius":"Density filter radius",
  "max_iter":"Maximum OC iterations","tolerance":"Convergence tolerance",
  "run":"Optimize structure","history":"Saved single-case solutions",
  "choose":"Select result","none":"No single-case results yet.",
  "compliance":"Final compliance","initial":"Initial compliance",
  "solves":"FEM calls","quality":"Physical checks","runtime":"Elapsed time (s)",
  "fields":"Density fields","cutoff":"Solid-density threshold",
  "layer":"Z-layer","download_vtk":"Download ParaView VTK",
  "download_npz":"Download density NPZ","download_json":"Download run summary",
  "run_busy":"A previous local job is active. Open Runs & logs to see progress.",
  "success":"Optimization started","error":"Cannot start FEM case",
  "load_warning":"Default 2D uses plane stress; 3D uses full spatial Hex8 elasticity. Both use a left clamped cantilever.",
  "choose_data":"Open stored results","convergence":"Convergence history",
 },
 "Русский":{
  "title":"Расчёт одной конструкции","description":"Реальная FEM/OC-оптимизация с заданными геометрией, сеткой, долей материала и положением нагрузки.",
  "nx":"Элементов X","ny":"Элементов Y","nz":"Элементов Z",
  "length":"Длина X","width":"Ширина Y","height":"Высота Z / высота пластины",
  "volume":"Допустимая доля материала","load_y":"Положение нагрузки Y",
  "load_z":"Положение нагрузки Z","radius":"Радиус фильтра плотности",
  "max_iter":"Максимум итераций OC","tolerance":"Порог сходимости",
  "run":"Оптимизировать конструкцию","history":"Сохранённые расчёты",
  "choose":"Выбрать результат","none":"Пока нет расчётов одной конструкции.",
  "compliance":"Итоговая податливость","initial":"Начальная податливость",
  "solves":"Число FEM-решений","quality":"Физические проверки","runtime":"Время расчёта (с)",
  "fields":"Поля плотности","cutoff":"Порог твёрдой плотности",
  "layer":"Слой Z","download_vtk":"Скачать VTK для ParaView",
  "download_npz":"Скачать плотность NPZ","download_json":"Скачать JSON-отчёт",
  "run_busy":"Другая задача уже выполняется. Открой «Запуски и журналы».",
  "success":"Расчёт запущен","error":"Не удалось запустить FEM-задачу",
  "load_warning":"В 2D используется plane stress, в 3D — полная пространственная упругость Hex8. Оба варианта — консоль с жёсткой левой опорой.",
  "choose_data":"Открыть сохранённые результаты","convergence":"История сходимости",
 }}
assert set(TEXT["English"])==set(TEXT["Русский"])


def show_density_3d(rho,case,t):
    import plotly.graph_objects as go
    nz,ny,nx=rho.shape
    threshold=st.slider(t("cutoff"),.05,.95,.45,step=.05,key="case_solid_threshold")
    zz,yy,xx=np.mgrid[:nz,:ny,:nx]
    active=rho>=threshold
    if np.any(active):
        values=rho[active]
        ix,iy,iz=xx[active],yy[active],zz[active]
        stride=max(1,int(np.ceil(len(values)/12000)))
        fig=go.Figure(data=[go.Scatter3d(
            x=(ix[::stride]+.5)*case["length"]/nx,
            y=(iy[::stride]+.5)*case["width"]/ny,
            z=(iz[::stride]+.5)*case["height"]/nz,
            mode="markers",
            marker=dict(size=max(2,100/max(nx,ny,nz)),color=values[::stride],
                        colorscale="Viridis",opacity=.85,cmin=0,cmax=1,
                        colorbar=dict(title="Density")))])
        fig.update_layout(scene=dict(aspectmode="data",xaxis_title="X",
                                     yaxis_title="Y",zaxis_title="Z"),
                          height=510,margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig,use_container_width=True)
    z_index=st.slider(t("layer"),0,nz-1,nz//2)
    st.image(np.flipud(np.uint8(rho[z_index]*255)),clamp=True,width=620)


def render(dimension,language):
    t=TEXT[language].__getitem__
    st.subheader(t("title")+" · "+dimension)
    st.caption(t("description"))
    st.info(t("load_warning"))
    has_active=bool(backend.active_runs())
    if has_active:
        st.warning(t("run_busy"))
    with st.form("manual_solver_"+dimension):
        a,b,c=st.columns(3)
        nx=a.number_input(t("nx"),value=16 if dimension=="2D" else 8,
                          min_value=2,max_value=512,step=2)
        ny=b.number_input(t("ny"),value=8 if dimension=="2D" else 4,
                          min_value=2,max_value=256,step=2)
        nz=c.number_input(t("nz"),value=4,min_value=2,max_value=128,
                          step=2,disabled=dimension=="2D")
        x1,x2,x3=st.columns(3)
        length=x1.number_input(t("length"),min_value=.05,max_value=100.,
                               value=2.,step=.1)
        width=x2.number_input(t("width"),min_value=.05,max_value=100.,
                              value=1.,step=.1,disabled=dimension=="2D")
        height=x3.number_input(t("height"),min_value=.05,max_value=100.,
                               value=1.,step=.1)
        d1,d2,d3=st.columns(3)
        volume=d1.slider(t("volume"),min_value=.1,max_value=.9,value=.5,step=.05)
        load_y=d2.slider(t("load_y"),min_value=.05,max_value=.95,value=.5,step=.05)
        load_z=d3.slider(t("load_z"),min_value=.05,max_value=.95,value=.5,
                         step=.05,disabled=dimension=="2D")
        r1,r2,r3=st.columns(3)
        radius=r1.number_input(t("radius"),min_value=.01,max_value=10.,
                               value=.4 if dimension=="3D" else .19,step=.01)
        max_iter=r2.number_input(t("max_iter"),min_value=15,max_value=10000,
                                value=400,step=25)
        tolerance=r3.number_input(t("tolerance"),min_value=.001,max_value=.5,
                                  value=.02,step=.005,format="%.3f")
        go=st.form_submit_button(t("run"),type="primary",disabled=has_active)
    if go:
        try:
            dim=2 if dimension=="2D" else 3
            options={"nx":int(nx),"ny":int(ny),"nz":int(nz),
                     "length":length,"width":width,"height":height,
                     "volume":volume,"load_y":load_y,"load_z":load_z,
                     "radius":radius,"max_iter":int(max_iter),"tolerance":tolerance}
            ident=backend.new_run(f"case_{dim}d",
                    lambda folder:[(f"{dim}D FEM + SIMP + OC",
                        backend.single_case_task(folder/"solution",dim,**options))],
                    {"dimension":dim,**options})
            st.success(f"{t('success')}: {ident}")
        except (ValueError,RuntimeError,OSError) as exc:
            st.error(f"{t('error')}: {exc}")
    st.divider()
    st.markdown("#### "+t("history"))
    dim=2 if dimension=="2D" else 3
    runs=[r for r in backend.list_runs()
          if r["kind"]==f"case_{dim}d" and
          (Path(r["directory"])/"solution"/"case.json").is_file()]
    if not runs:
        st.info(t("none"))
        return
    run=st.selectbox(t("choose"),runs,
                     format_func=lambda r:r["id"]+" / "+r["status"])
    folder=Path(run["directory"])/"solution"
    summary=json.loads((folder/"case.json").read_text(encoding="utf-8"))
    a,b,c,d=st.columns(4)
    a.metric(t("initial"),f'{summary["initial_compliance"]:.5g}')
    b.metric(t("compliance"),f'{summary["compliance"]:.5g}')
    c.metric(t("solves"),summary["fe_solves"])
    d.metric(t("runtime"),f'{summary["seconds"]:.3f}')
    st.json(summary["quality_checks"],expanded=False)
    density_file=folder/"density.npz"
    if density_file.stat().st_size > 256*1024*1024:
        st.warning("Large result: download NPZ for external postprocessing")
    else:
        with np.load(density_file,allow_pickle=False) as npz:
            rho=npz["physical_density"]
        st.markdown("#### "+t("fields"))
        if dim==3:
            show_density_3d(rho,summary["case"],t)
        else:
            st.image(np.flipud(np.uint8(rho*255)),clamp=True,width=720)
    st.markdown("#### "+t("convergence"))
    history=pd.read_csv(folder/"history.csv")
    if not history.empty:
        st.line_chart(history.set_index("iteration")[["compliance"]])
        with st.expander(t("convergence")):
            st.dataframe(history,hide_index=True)
    st.download_button(t("download_npz"),density_file.read_bytes(),
                       file_name=f"density_{dim}d.npz")
    st.download_button(t("download_json"),(folder/"case.json").read_bytes(),
                       file_name="case.json")
    if dim==3 and (folder/"density.vtk").is_file():
        st.download_button(t("download_vtk"),(folder/"density.vtk").read_bytes(),
                           file_name="density.vtk")
