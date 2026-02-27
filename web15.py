import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

if not hasattr(np, 'bool'):
    np.bool = bool

def setup_chinese_font():
    """设置中文字体（云端优先加载本地fonts目录内的CJK字体）"""
    try:
        import os
        import matplotlib.font_manager as fm

        # 优先尝试系统已安装字体
        chinese_fonts = [
            'WenQuanYi Zen Hei',
            'WenQuanYi Micro Hei',
            'SimHei',
            'Microsoft YaHei',
            'PingFang SC',
            'Hiragino Sans GB',
            'Noto Sans CJK SC',
            'Source Han Sans SC'
        ]

        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in chinese_fonts:
            if font in available_fonts:
                matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                print(f"使用中文字体: {font}")
                return font

        # 若系统无中文字体，尝试从./fonts 目录加载随应用打包的字体
        candidates = [
            'NotoSansSC-Regular.otf',
            'NotoSansCJKsc-Regular.otf',
            'SourceHanSansSC-Regular.otf',
            'SimHei.ttf',
            'MicrosoftYaHei.ttf'
        ]
        fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        if os.path.isdir(fonts_dir):
            for fname in candidates:
                fpath = os.path.join(fonts_dir, fname)
                if os.path.exists(fpath):
                    try:
                        fm.fontManager.addfont(fpath)
                        fp = fm.FontProperties(fname=fpath)
                        fam = fp.get_name()
                        matplotlib.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans', 'Arial']
                        matplotlib.rcParams['font.family'] = 'sans-serif'
                        print(f"使用本地打包字体: {fam} ({fname})")
                        return fam
                    except Exception as ie:
                        print(f"加载本地字体失败 {fname}: {ie}")

        # 兜底：使用英文字体（中文将显示为方框）
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        print("未找到中文字体，使用默认英文字体")
        return None

    except Exception as e:
        print(f"字体设置失败: {e}")
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        return None

chinese_font = setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False # 确保可以显示负号

# ==============================================================================
# 1. 项目名称和配置 
# ==============================================================================
st.set_page_config(
    page_title="基于机器学习模型的结直肠癌化疗相关骨髓抑制风险预测",
    page_icon="🧬", 
    layout="wide"
)

if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'Arial']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False 


global feature_names_display, feature_dict, variable_descriptions


# 14个特征（注意大小写与文档一致）
feature_names_display = [
    'tumor_stage_advanced',  # 肿瘤分期（≥III期）
    'Liver_metastasis',      # 肝转移
    'Lung_metastasis',       # 肺转移
    'Peritoneal_metastasis', # 腹膜转移
    'Age',                   # 年龄
    'BMI',                   # BMI
    'CEA',                   # 癌胚抗原
    'WBC',                   # 白细胞
    'ANC',                   # 中性粒细胞计数
    'PLT',                   # 血小板计数
    'ALB',                   # 白蛋白
    'TP',                    # 总蛋白
    'CRP',                   # C反应蛋白
    'eGFR'                   # 估算肾小球滤过率
]

# 14个特征的中文名称
feature_names_cn = [
    '肿瘤分期（≥III期）', '肝转移', '肺转移', '腹膜转移',
    '年龄', 'BMI', '癌胚抗原（CEA）', '白细胞（WBC）',
    '中性粒细胞计数（ANC）', '血小板计数（PLT）', '白蛋白（ALB）', '总蛋白（TP）',
    'C反应蛋白（CRP）', '估算肾小球滤过率（eGFR）'
]

# 用于英文键名到中文显示名的映射
feature_dict = dict(zip(feature_names_display, feature_names_cn))

# 变量说明字典：键名与模型要求的格式一致（注意大小写）
variable_descriptions = {
    'tumor_stage_advanced': '是否有肿瘤分期≥III期（0=无，1=有）',
    'Liver_metastasis': '是否有肝转移（0=无，1=有）',
    'Lung_metastasis': '是否有肺转移（0=无，1=有）',
    'Peritoneal_metastasis': '是否有腹膜转移（0=无，1=有）',
    'Age': '年龄（岁）',
    'BMI': '体重指数（kg/m²）',
    'CEA': '癌胚抗原（ng/mL）',
    'WBC': '白细胞计数（×10⁹/L）',
    'ANC': '中性粒细胞计数（×10⁹/L）',
    'PLT': '血小板计数（×10⁹/L）',
    'ALB': '白蛋白（g/L）',
    'TP': '总蛋白（g/L）',
    'CRP': 'C反应蛋白（mg/L）',
    'eGFR': '估算肾小球滤过率（mL/min/1.73m²）'
}

@st.cache_resource
def load_model(model_path: str = './xgb_model.pkl'):
    """加载模型文件，优先使用joblib，其次pickle"""
    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        # 尝试获取模型内部特征名
        model_feature_names = None
        if hasattr(model, 'feature_names_in_'):
            model_feature_names = list(model.feature_names_in_)
        else:
            try:
                # 针对XGBoost/LightGBM等尝试获取booster
                booster = getattr(model, 'get_booster', lambda: None)()
                if booster is not None:
                    model_feature_names = booster.feature_names
            except Exception:
                model_feature_names = None

        return model, model_feature_names
    except Exception as e:
        raise RuntimeError(f"无法加载模型，请检查文件路径和格式: {e}")


def main():
    global feature_names_display, feature_dict, variable_descriptions

    # ==============================================================================
    # 2. 侧边栏和主标题 
    # ==============================================================================
    # 侧边栏标题
    st.sidebar.title("结直肠癌化疗相关骨髓抑制风险预测模型")
    st.sidebar.image("https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg", width=200) 

    # 添加系统说明到侧边栏
    st.sidebar.markdown("""
    # 系统说明

    ## 关于本系统
    这是一个基于机器学习算法的**结直肠癌化疗相关骨髓抑制**风险预测系统，用于评估患者发生骨髓抑制的风险。

    ## 预测结果
    系统输出：
    - **骨髓抑制**发生概率
    - 未发生**骨髓抑制**概率
    - 风险分层（低/中/高）
    """)

    # 添加变量说明到侧边栏
    with st.sidebar.expander("变量说明"):
        for feature in feature_names_display:
            st.markdown(f"**{feature_dict.get(feature, feature)}**: {variable_descriptions.get(feature, '无详细说明')}")


    # 主页面标题
    st.title("基于机器学习模型的结直肠癌化疗相关骨髓抑制风险预测")
    st.markdown("### 请在下方录入全部特征后进行预测")

    # 加载模型
    try:
        model, model_feature_names = load_model('./xgb_model.pkl')
        st.sidebar.success("模型加载成功！")
    except Exception as e:
        st.sidebar.error(f"模型加载失败: {e}")
        return


    # ==============================================================================
    # 3. 特征输入控件 - 使用4列布局容纳14个特征
    # ==============================================================================
    st.header("患者指标录入")
    # 使用 4 列布局来容纳 14 个特征 (4+4+4+2=14)
    col1, col2, col3, col4 = st.columns(4) 
    
    # 类别变量的格式化函数
    to_cn = lambda x: "有" if x == 1 else "无"

    # --- 第 1 列 (特征 1-4，二分类变量) ---
    with col1:
        # 1. 肿瘤分期（≥III期）（0/1）
        tumor_stage_advanced = st.selectbox("肿瘤分期（≥III期）", options=[0, 1], format_func=to_cn, index=0, key='tumor_stage') 
        # 2. 肝转移（0/1）
        liver_metastasis = st.selectbox("肝转移", options=[0, 1], format_func=to_cn, index=0, key='liver') 
        # 3. 肺转移（0/1）
        lung_metastasis = st.selectbox("肺转移", options=[0, 1], format_func=to_cn, index=0, key='lung')
        # 4. 腹膜转移（0/1）
        peritoneal_metastasis = st.selectbox("腹膜转移", options=[0, 1], format_func=to_cn, index=0, key='peritoneal')

    # --- 第 2 列 (特征 5-8) ---
    with col2:
        # 5. 年龄（数值）
        age = st.number_input("年龄（岁）", value=60, step=1, min_value=18, max_value=120, key='age_val') 
        # 6. BMI（数值）
        bmi = st.number_input("BMI（kg/m²）", value=22.0, step=0.1, min_value=10.0, max_value=50.0, key='bmi')
        # 7. 癌胚抗原（CEA）（数值）
        cea = st.number_input("癌胚抗原（ng/mL）", value=5.0, step=0.1, min_value=0.0, key='cea')
        # 8. 白细胞（WBC）（数值）
        wbc = st.number_input("白细胞（×10⁹/L）", value=6.0, step=0.1, min_value=0.0, key='wbc')

    # --- 第 3 列 (特征 9-12) ---
    with col3:
        # 9. 中性粒细胞计数（ANC）（数值）
        anc = st.number_input("中性粒细胞计数（×10⁹/L）", value=3.5, step=0.1, min_value=0.0, key='anc')
        # 10. 血小板计数（PLT）（数值）
        plt_val = st.number_input("血小板计数（×10⁹/L）", value=200.0, step=1.0, min_value=0.0, key='plt')
        # 11. 白蛋白（ALB）（数值）
        alb = st.number_input("白蛋白（g/L）", value=40.0, step=0.1, min_value=0.0, key='alb')
        # 12. 总蛋白（TP）（数值）
        tp = st.number_input("总蛋白（g/L）", value=70.0, step=0.1, min_value=0.0, key='tp')

    # --- 第 4 列 (特征 13-14) ---
    with col4:
        # 13. C反应蛋白（CRP）（数值）
        crp = st.number_input("C反应蛋白（mg/L）", value=5.0, step=0.1, min_value=0.0, key='crp')
        # 14. 估算肾小球滤过率（eGFR）（数值）
        egfr = st.number_input("估算肾小球滤过率（mL/min/1.73m²）", value=90.0, step=0.1, min_value=0.0, key='egfr')


    # 预测按钮
    predict_button = st.button("开始预测", type="primary")

    if predict_button:
        # 根据模型的特征顺序构建输入DataFrame（注意大小写与feature_names_display一致）
        user_inputs = {
            'tumor_stage_advanced': tumor_stage_advanced,
            'Liver_metastasis': liver_metastasis,
            'Lung_metastasis': lung_metastasis,
            'Peritoneal_metastasis': peritoneal_metastasis,
            'Age': age,
            'BMI': bmi,
            'CEA': cea,
            'WBC': wbc,
            'ANC': anc,
            'PLT': plt_val,
            'ALB': alb,
            'TP': tp,
            'CRP': crp,
            'eGFR': egfr,
        }

        # 特征对齐逻辑
        if model_feature_names:
            # 简化特征名映射（假设模型特征名与 feature_names_display 相似）
            alias_to_user_key = {f: f for f in feature_names_display}
            
            resolved_values = []
            missing_features = []
            for c in model_feature_names: # 遍历模型要求的特征名
                ui_key = alias_to_user_key.get(c, c) 
                val = user_inputs.get(ui_key, user_inputs.get(c, None)) 
                if val is None:
                    missing_features.append(c)
                resolved_values.append(val)

            if missing_features:
                st.error(f"以下模型特征未在页面录入或名称不匹配：{missing_features}。\n请核对特征名（注意大小写）。")
                with st.expander("调试信息：模型与输入特征名对比"):
                    st.write("模型特征名：", model_feature_names)
                    st.write("页面输入键：", list(user_inputs.keys()))
                return

            input_df = pd.DataFrame([resolved_values], columns=model_feature_names)
        else:
            # 如果无法获取模型特征名，则使用 feature_names_display 顺序
            ordered_cols = feature_names_display
            input_df = pd.DataFrame([[user_inputs[c] for c in ordered_cols]], columns=ordered_cols)

        # 简单检查缺失
        if input_df.isnull().any().any():
            st.error("存在缺失的输入值，请完善后重试。")
            return

        # 确保 input_df 中的数据类型为数字
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            except Exception:
                pass

        # 进行预测（概率）
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)[0]
                # 假设第1列为阴性（未发生），第2列为阳性（发生）
                if len(proba) == 2:
                    no_mys_prob = float(proba[0])
                    mys_prob = float(proba[1]) # 骨髓抑制发生概率
                else:
                    raise ValueError("predict_proba返回的维度异常")
            else:
                # 预测失败的退路，概率近似
                if hasattr(model, 'decision_function'):
                    score = float(model.decision_function(input_df))
                    mys_prob = 1 / (1 + np.exp(-score))
                    no_mys_prob = 1 - mys_prob
                else:
                    pred = int(model.predict(input_df)[0])
                    mys_prob = float(pred)
                    no_mys_prob = 1 - mys_prob

            # 显示预测结果
            st.header("化疗相关骨髓抑制风险预测结果")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("未发生骨髓抑制概率")
                st.progress(no_mys_prob) 
                st.write(f"{no_mys_prob:.2%}")
            with col2:
                st.subheader("骨髓抑制发生概率")
                st.progress(mys_prob) 
                st.write(f"{mys_prob:.2%}")

            # 风险分层
            risk_level = "低风险" if mys_prob < 0.3 else ("中等风险" if mys_prob < 0.7 else "高风险")
            risk_color = "green" if mys_prob < 0.3 else ("orange" if mys_prob < 0.7 else "red")
            st.markdown(f"### 骨髓抑制风险评估: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
            
            # ====== 诊疗建议 ======
            st.write("---")
            st.header("诊疗建议")
            
            if mys_prob < 0.3:
                st.markdown("#### 低风险")
                st.info("建议采用标准剂量化疗方案。定期监测血常规（建议每2-3周一次），关注白细胞、中性粒细胞及血小板计数变化。加强营养支持，维持良好的一般状况。")
            elif mys_prob < 0.7:
                st.markdown("#### 中等风险")
                st.warning("建议加强化疗期间监测，考虑预防性使用粒细胞集落刺激因子（G-CSF）。缩短血常规复查间隔（每1-2周一次）。若出现骨髓抑制迹象，及时调整化疗药物剂量或延迟化疗。注意肝肾功能保护，积极处理转移灶相关并发症。")
            else:
                st.markdown("#### 高风险")
                st.error("强烈建议预防性使用G-CSF支持治疗，考虑降低化疗药物起始剂量或选择骨髓毒性较低的方案。每周监测血常规，必要时住院治疗。若发生严重骨髓抑制（III-IV级），应立即停止化疗并给予积极支持治疗（包括输血、抗感染等）。多学科会诊评估患者是否适合继续化疗。")
            # ==========================

        except Exception as e:
            st.error(f"预测或结果展示失败: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    # 版权或说明
    st.write("---")
    st.caption("© 2026 基于机器学习的结直肠癌化疗相关骨髓抑制风险预测模型")

if __name__ == "__main__":
    main()