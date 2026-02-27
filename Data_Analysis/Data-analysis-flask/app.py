from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import json

app = Flask(__name__)
np.random.seed(42)

# ─── HOME ──────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ═══════════════════════════════════════════════════════════════
# 01 — NUMPY
# ═══════════════════════════════════════════════════════════════
@app.route("/numpy")
def numpy_page():
    return render_template("numpy/index.html")

@app.route("/api/numpy/array", methods=["POST"])
def api_numpy_array():
    data   = request.json
    values = [float(x.strip()) for x in data.get("values","1,2,3,4,5").split(",") if x.strip()]
    arr    = np.array(values)
    return jsonify({
        "array":   arr.tolist(),
        "shape":   list(arr.shape),
        "dtype":   str(arr.dtype),
        "ndim":    int(arr.ndim),
        "size":    int(arr.size),
    })

@app.route("/api/numpy/stats", methods=["POST"])
def api_numpy_stats():
    data   = request.json
    values = [float(x.strip()) for x in data.get("values","").split(",") if x.strip()]
    arr    = np.array(values)
    return jsonify({
        "mean":      round(float(np.mean(arr)), 4),
        "median":    round(float(np.median(arr)), 4),
        "std":       round(float(np.std(arr)), 4),
        "variance":  round(float(np.var(arr)), 4),
        "min":       round(float(np.min(arr)), 4),
        "max":       round(float(np.max(arr)), 4),
        "pct25":     round(float(np.percentile(arr, 25)), 4),
        "pct75":     round(float(np.percentile(arr, 75)), 4),
        "iqr":       round(float(np.percentile(arr,75) - np.percentile(arr,25)), 4),
        "sum":       round(float(np.sum(arr)), 4),
    })

@app.route("/api/numpy/mask", methods=["POST"])
def api_numpy_mask():
    data      = request.json
    values    = [float(x.strip()) for x in data.get("values","").split(",") if x.strip()]
    threshold = float(data.get("threshold", 50))
    arr       = np.array(values)
    above     = arr[arr >= threshold].tolist()
    below     = arr[arr <  threshold].tolist()
    return jsonify({"above": above, "below": below,
                    "above_count": len(above), "below_count": len(below),
                    "above_pct": round(len(above)/len(arr)*100, 1)})

@app.route("/api/numpy/random", methods=["POST"])
def api_numpy_random():
    data = request.json
    n    = int(data.get("n", 100))
    mean = float(data.get("mean", 65))
    std  = float(data.get("std", 15))
    np.random.seed(42)
    arr  = np.random.normal(mean, std, n).clip(0, 100).round(1)
    return jsonify({
        "sample":   arr[:10].tolist(),
        "mean":     round(float(arr.mean()), 2),
        "std":      round(float(arr.std()), 2),
        "min":      round(float(arr.min()), 2),
        "max":      round(float(arr.max()), 2),
        "pass_pct": round(float((arr >= 60).mean() * 100), 1),
    })

# ═══════════════════════════════════════════════════════════════
# 02 — PANDAS
# ═══════════════════════════════════════════════════════════════
@app.route("/pandas")
def pandas_page():
    return render_template("pandas/index.html")

def make_student_df(n=50):
    np.random.seed(42)
    df = pd.DataFrame({
        "student_id":    range(1001, 1001+n),
        "name":          [f"Student_{i}" for i in range(n)],
        "age":           np.random.randint(17, 25, n).tolist(),
        "gender":        np.random.choice(["Male","Female"], n).tolist(),
        "department":    np.random.choice(["CS","Maths","Physics","Chemistry"], n).tolist(),
        "math_score":    np.random.randint(40, 100, n).tolist(),
        "science_score": np.random.randint(35, 100, n).tolist(),
        "english_score": np.random.randint(30, 100, n).tolist(),
    })
    df["avg"] = ((df["math_score"] + df["science_score"] + df["english_score"]) / 3).round(2)
    df["passed"] = (df["avg"] >= 60).astype(bool)
    return df

@app.route("/api/pandas/preview", methods=["GET"])
def api_pandas_preview():
    df = make_student_df()
    return jsonify({
        "rows":    df.head(10).to_dict("records"),
        "shape":   list(df.shape),
        "columns": df.columns.tolist(),
    })

@app.route("/api/pandas/groupby", methods=["POST"])
def api_pandas_groupby():
    df   = make_student_df()
    col  = request.json.get("col", "department")
    grp  = df.groupby(col).agg(
        count   =("student_id","count"),
        avg_math=("math_score","mean"),
        avg_score=("avg","mean"),
        pass_rate=("passed","mean"),
    ).round(2)
    grp["pass_rate"] = (grp["pass_rate"]*100).round(1)
    return jsonify(grp.reset_index().to_dict("records"))

@app.route("/api/pandas/filter", methods=["POST"])
def api_pandas_filter():
    df   = make_student_df()
    data = request.json
    dept = data.get("department", "CS")
    min_avg = float(data.get("min_avg", 60))
    result  = df[(df["department"] == dept) & (df["avg"] >= min_avg)]
    return jsonify({
        "rows":  result.head(10).to_dict("records"),
        "count": len(result),
    })

# ═══════════════════════════════════════════════════════════════
# 03 — DATA CLEANING
# ═══════════════════════════════════════════════════════════════
@app.route("/cleaning")
def cleaning_page():
    return render_template("cleaning/index.html")

def make_dirty_df():
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "emp_id":     list(range(1001,1001+n)) + [1010,1025],
        "name":       [f"Emp_{i}" for i in range(n)] + ["Emp_9","Emp_24"],
        "age":        list(np.random.randint(22,60,n)) + [22,35],
        "department": list(np.random.choice(["IT","HR","Finance","IT ","hr","FINANCE"], n))
                      + ["IT","HR"],
        "salary":     list(np.random.normal(55000,15000,n).round(0)) + [55000,48000],
        "rating":     list(np.random.choice([1,2,3,4,5,None,999,-1], n)) + [3,4],
    })
    for col in ["salary","age"]:
        mask = np.random.choice([True,False], len(df), p=[0.08,0.92])
        df.loc[mask, col] = np.nan
    df.loc[5,  "salary"] = 950000
    df.loc[15, "salary"] = -5000
    df.loc[30, "age"]    = 135
    return df

@app.route("/api/cleaning/audit", methods=["GET"])
def api_cleaning_audit():
    df = make_dirty_df()
    missing = df.isnull().sum().to_dict()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(1).to_dict()
    return jsonify({
        "shape":       list(df.shape),
        "duplicates":  int(df.duplicated(subset="emp_id").sum()),
        "missing":     missing,
        "missing_pct": missing_pct,
        "dept_unique": df["department"].unique().tolist(),
        "sample":      df.head(8).fillna("NULL").to_dict("records"),
    })

@app.route("/api/cleaning/clean", methods=["GET"])
def api_cleaning_clean():
    df = make_dirty_df()
    before_shape = df.shape

    # Step 1 — duplicates
    df = df.drop_duplicates(subset="emp_id", keep="first")
    # Step 2 — standardize
    df["department"] = df["department"].str.strip().str.title().replace({"It":"IT","Hr":"HR"})
    # Step 3 — invalid values
    df.loc[df["age"]>80,    "age"]    = np.nan
    df.loc[df["salary"]<0,  "salary"] = np.nan
    Q1 = df["salary"].quantile(0.25); Q3 = df["salary"].quantile(0.75)
    cap = Q3 + 3*(Q3-Q1)
    df.loc[df["salary"]>cap, "salary"] = cap
    df.loc[~df["rating"].isin([1,2,3,4,5]), "rating"] = np.nan
    # Step 4 — fill
    df["age"]    = df["age"].fillna(df["age"].median())
    df["salary"] = df.groupby("department")["salary"].transform(lambda x: x.fillna(x.median()))
    df["rating"] = df["rating"].fillna(df["rating"].mode()[0])
    df["age"]    = df["age"].astype(int)
    df["salary"] = df["salary"].round(2)
    df["rating"] = df["rating"].astype(int)

    return jsonify({
        "before_shape": list(before_shape),
        "after_shape":  list(df.shape),
        "missing_after": int(df.isnull().sum().sum()),
        "dept_unique":  df["department"].unique().tolist(),
        "salary_max":   float(df["salary"].max()),
        "sample":       df.head(8).to_dict("records"),
    })

# ═══════════════════════════════════════════════════════════════
# 04 — EDA & DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════
@app.route("/eda")
def eda_page():
    return render_template("eda/index.html")

def make_eda_df():
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "age":        np.random.normal(35,8,n).clip(18,65).round().astype(int),
        "salary":     np.random.lognormal(10.8,0.4,n).round(0).clip(25000,200000),
        "experience": np.random.randint(0,30,n),
        "department": np.random.choice(["IT","HR","Finance","Marketing","Operations"],n,
                                        p=[.30,.15,.20,.20,.15]),
        "gender":     np.random.choice(["Male","Female"],n,p=[.55,.45]),
        "rating":     np.random.choice([1,2,3,4,5],n,p=[.05,.10,.25,.40,.20]),
        "education":  np.random.choice(["Bachelor's","Master's","PhD"],n,p=[.55,.35,.10]),
    })
    return df

@app.route("/api/eda/univariate", methods=["POST"])
def api_eda_univariate():
    df  = make_eda_df()
    col = request.json.get("col", "salary")
    s   = df[col]
    return jsonify({
        "mean":    round(float(s.mean()), 2),
        "median":  round(float(s.median()), 2),
        "std":     round(float(s.std()), 2),
        "skew":    round(float(s.skew()), 3),
        "kurtosis":round(float(s.kurtosis()), 3),
        "min":     round(float(s.min()), 2),
        "max":     round(float(s.max()), 2),
        "hist":    list(np.histogram(s, bins=15)[0].tolist()),
        "bins":    [round(x,1) for x in np.histogram(s, bins=15)[1].tolist()],
        "skew_label": "Right-skewed" if s.skew()>0.5 else ("Left-skewed" if s.skew()<-0.5 else "Symmetric"),
    })

@app.route("/api/eda/categorical", methods=["POST"])
def api_eda_categorical():
    df  = make_eda_df()
    col = request.json.get("col","department")
    vc  = df[col].value_counts()
    return jsonify({"labels": vc.index.tolist(), "values": vc.values.tolist()})

@app.route("/api/eda/correlation", methods=["GET"])
def api_eda_correlation():
    df   = make_eda_df()
    cols = ["age","salary","experience","rating"]
    corr = df[cols].corr().round(3)
    return jsonify({"cols": cols, "matrix": corr.values.tolist()})

@app.route("/api/eda/bivariate", methods=["POST"])
def api_eda_bivariate():
    df = make_eda_df()
    x_col = request.json.get("x","experience")
    y_col = request.json.get("y","salary")
    m, b, r, p, _ = stats.linregress(df[x_col], df[y_col])
    sample = df[[x_col, y_col]].sample(80, random_state=42)
    return jsonify({
        "x": sample[x_col].tolist(),
        "y": sample[y_col].tolist(),
        "r": round(r, 3),
        "slope": round(m, 3),
    })

# ═══════════════════════════════════════════════════════════════
# 05 — OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════
@app.route("/outliers")
def outliers_page():
    return render_template("outliers/index.html")

def make_outlier_df():
    np.random.seed(42)
    n = 200
    salary = np.random.normal(60000,12000,n).tolist()
    salary += [200000, 220000, -5000, 185000, 3000]
    age    = np.random.normal(35,8,n).clip(18,65).tolist()
    age   += [120, 130, 2, 1, 125]
    return pd.DataFrame({"salary": salary, "age": age})

@app.route("/api/outliers/detect", methods=["POST"])
def api_outliers_detect():
    df     = make_outlier_df()
    method = request.json.get("method","iqr")
    col    = request.json.get("col","salary")
    s      = df[col]

    if method == "iqr":
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR    = Q3 - Q1
        low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        mask   = (s < low) | (s > high)
        bounds = {"lower": round(low,2), "upper": round(high,2)}
    elif method == "zscore":
        z    = np.abs(stats.zscore(s))
        mask = z > 3
        bounds = {"threshold": 3}
    else:  # modified z
        median = s.median()
        mad    = np.median(np.abs(s - median))
        mz     = 0.6745 * (s - median) / (mad + 1e-10)
        mask   = np.abs(mz) > 3.5
        bounds = {"threshold": 3.5}

    return jsonify({
        "total":          len(s),
        "outlier_count":  int(mask.sum()),
        "outlier_pct":    round(mask.mean()*100, 1),
        "outlier_values": sorted(s[mask].round(1).tolist()),
        "bounds":         bounds,
        "normal_sample":  s[~mask].sample(min(20,int((~mask).sum())), random_state=1).round(1).tolist(),
    })

@app.route("/api/outliers/treat", methods=["POST"])
def api_outliers_treat():
    df        = make_outlier_df()
    treatment = request.json.get("treatment","cap")
    s         = df["salary"].copy()
    Q1, Q3    = s.quantile(0.25), s.quantile(0.75)
    IQR       = Q3 - Q1
    low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR

    if treatment == "remove":
        mask  = (s >= low) & (s <= high)
        after = s[mask]
    elif treatment == "cap":
        after = s.clip(lower=low, upper=high)
    else:  # log
        after = np.log1p(s[s > 0])

    return jsonify({
        "before_count": len(s),
        "after_count":  len(after),
        "before_skew":  round(float(s.skew()), 3),
        "after_skew":   round(float(after.skew()), 3),
        "before_max":   round(float(s.max()), 1),
        "after_max":    round(float(after.max()), 1),
        "sample":       after.round(1).tolist()[:20],
    })

# ═══════════════════════════════════════════════════════════════
# 06 — DATA WRANGLING
# ═══════════════════════════════════════════════════════════════
@app.route("/wrangling")
def wrangling_page():
    return render_template("wrangling/index.html")

def make_wrangling_dfs():
    np.random.seed(42)
    employees = pd.DataFrame({
        "emp_id":    range(1001,1016),
        "name":      [f"Emp_{i}" for i in range(15)],
        "dept_id":   np.random.choice([10,20,30,40], 15).tolist(),
        "salary":    np.random.randint(35000,120000,15).tolist(),
        "hire_year": np.random.choice([2019,2020,2021,2022,2023], 15).tolist(),
    })
    departments = pd.DataFrame({
        "dept_id":  [10,20,30,40],
        "dept_name":["IT","HR","Finance","Marketing"],
        "location": ["Bangalore","Mumbai","Delhi","Chennai"],
    })
    performance = pd.DataFrame({
        "emp_id":   range(1001,1016),
        "q1_score": np.random.randint(50,100,15).tolist(),
        "q2_score": np.random.randint(50,100,15).tolist(),
        "q3_score": np.random.randint(50,100,15).tolist(),
        "q4_score": np.random.randint(50,100,15).tolist(),
    })
    return employees, departments, performance

@app.route("/api/wrangling/merge", methods=["POST"])
def api_wrangling_merge():
    emp, dept, _ = make_wrangling_dfs()
    how  = request.json.get("how","inner")
    merged = emp.merge(dept, on="dept_id", how=how)
    return jsonify({
        "rows":  merged.head(10).fillna("NULL").to_dict("records"),
        "count": len(merged),
        "cols":  merged.columns.tolist(),
    })

@app.route("/api/wrangling/melt", methods=["GET"])
def api_wrangling_melt():
    _, _, perf = make_wrangling_dfs()
    wide = perf.head(5)
    long = wide.melt(id_vars="emp_id",
                     value_vars=["q1_score","q2_score","q3_score","q4_score"],
                     var_name="quarter", value_name="score")
    long["quarter"] = long["quarter"].str.replace("_score","").str.upper()
    return jsonify({
        "wide": wide.to_dict("records"),
        "long": long.to_dict("records"),
    })

@app.route("/api/wrangling/cut", methods=["POST"])
def api_wrangling_cut():
    emp, _, _ = make_wrangling_dfs()
    ctype     = request.json.get("type","cut")
    if ctype == "cut":
        emp["band"] = pd.cut(emp["salary"],
                              bins=[0,50000,70000,90000,200000],
                              labels=["Low","Medium","High","Very High"])
        vc = emp["band"].value_counts().sort_index()
    else:
        emp["quartile"] = pd.qcut(emp["salary"], q=4, labels=["Q1","Q2","Q3","Q4"])
        vc = emp["quartile"].value_counts().sort_index()
    return jsonify({"labels": vc.index.astype(str).tolist(), "values": vc.values.tolist()})

@app.route("/api/wrangling/transform", methods=["GET"])
def api_wrangling_transform():
    emp, dept, _ = make_wrangling_dfs()
    full = emp.merge(dept, on="dept_id", how="left")
    full["dept_avg"]  = full.groupby("dept_name")["salary"].transform("mean").round(0)
    full["vs_avg"]    = (full["salary"] - full["dept_avg"]).round(0)
    full["dept_rank"] = full.groupby("dept_name")["salary"].rank(ascending=False).astype(int)
    return jsonify(full[["name","dept_name","salary","dept_avg","vs_avg","dept_rank"]].to_dict("records"))

# ═══════════════════════════════════════════════════════════════
# 07 — FEATURE ENGINEERING & SELECTION
# ═══════════════════════════════════════════════════════════════
@app.route("/features")
def features_page():
    return render_template("features/index.html")

def make_features_df():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age":          np.random.randint(22,60,n),
        "salary":       np.random.lognormal(10.8,0.4,n).round(0).clip(25000,200000),
        "experience":   np.random.randint(0,35,n),
        "num_projects": np.random.randint(1,20,n),
        "department":   np.random.choice(["IT","HR","Finance","Marketing"],n),
        "education":    np.random.choice(["Bachelor's","Master's","PhD"],n),
        "gender":       np.random.choice(["Male","Female"],n),
    })
    log_odds = 0.03*df["age"] + 0.00001*df["salary"] + 0.05*df["experience"] + 0.1*df["num_projects"] - 3
    df["promoted"] = (np.random.rand(n) < 1/(1+np.exp(-log_odds))).astype(int)
    return df

@app.route("/api/features/engineer", methods=["GET"])
def api_features_engineer():
    df = make_features_df()
    df["salary_log"]         = np.log1p(df["salary"]).round(4)
    df["experience_squared"] = (df["experience"] ** 2).astype(int)
    df["exp_per_age"]        = (df["experience"] / df["age"]).round(4)
    df["salary_per_project"] = (df["salary"] / df["num_projects"]).round(2)
    df["age_group"] = pd.cut(df["age"], bins=[18,30,40,50,65],
                              labels=["Junior","Mid","Senior","Expert"]).astype(str)
    df["salary_tier"] = pd.qcut(df["salary"], q=4, labels=["Q1","Q2","Q3","Q4"]).astype(str)
    new_cols = ["age","salary","salary_log","experience","experience_squared",
                "exp_per_age","salary_per_project","age_group","salary_tier","promoted"]
    return jsonify(df[new_cols].head(10).to_dict("records"))

@app.route("/api/features/encode", methods=["POST"])
def api_features_encode():
    df     = make_features_df().head(10)
    method = request.json.get("method","label")
    result = df[["gender","department","education"]].copy()
    if method == "label":
        le = LabelEncoder()
        result["gender_encoded"] = le.fit_transform(df["gender"])
        return jsonify(result[["gender","gender_encoded"]].to_dict("records"))
    elif method == "ordinal":
        result["education_ordinal"] = df["education"].map({"Bachelor's":0,"Master's":1,"PhD":2})
        return jsonify(result[["education","education_ordinal"]].to_dict("records"))
    else:  # ohe
        ohe = pd.get_dummies(df["department"], prefix="dept", drop_first=True)
        out = pd.concat([df[["department"]], ohe], axis=1)
        return jsonify(out.to_dict("records"))

@app.route("/api/features/scale", methods=["POST"])
def api_features_scale():
    df     = make_features_df()
    method = request.json.get("method","minmax")
    col    = request.json.get("col","salary")
    values = df[[col]].values

    scalers = {"minmax": MinMaxScaler(), "standard": StandardScaler(), "robust": RobustScaler()}
    scaler  = scalers.get(method, MinMaxScaler())
    scaled  = scaler.fit_transform(values).flatten()

    return jsonify({
        "original_sample": [round(v,2) for v in values.flatten()[:10].tolist()],
        "scaled_sample":   [round(v,4) for v in scaled[:10].tolist()],
        "scaled_min":      round(float(scaled.min()),4),
        "scaled_max":      round(float(scaled.max()),4),
        "scaled_mean":     round(float(scaled.mean()),4),
    })

@app.route("/api/features/selection", methods=["POST"])
def api_features_selection():
    df     = make_features_df()
    method = request.json.get("method","rf")

    df["gender_code"] = LabelEncoder().fit_transform(df["gender"])
    df["dept_code"]   = LabelEncoder().fit_transform(df["department"])
    df["edu_code"]    = df["education"].map({"Bachelor's":0,"Master's":1,"PhD":2})

    X = df[["age","salary","experience","num_projects","gender_code","dept_code","edu_code"]].fillna(0)
    y = df["promoted"]
    features = X.columns.tolist()

    if method == "rf":
        rf  = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        imp = dict(zip(features, [round(float(v),4) for v in rf.feature_importances_]))
        result = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        return jsonify({"method":"Random Forest Importance","result": [{"feature":k,"score":v} for k,v in result]})

    elif method == "kbest":
        skb = SelectKBest(f_classif, k=5)
        skb.fit(X, y)
        scores   = dict(zip(features, [round(float(v),2) for v in skb.scores_]))
        selected = [features[i] for i in range(len(features)) if skb.get_support()[i]]
        result   = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return jsonify({"method":"SelectKBest (f_classif)","selected": selected,
                        "result": [{"feature":k,"score":v} for k,v in result]})

    elif method == "rfe":
        rfe = RFE(LogisticRegression(max_iter=500), n_features_to_select=5)
        rfe.fit(X, y)
        selected = [features[i] for i in range(len(features)) if rfe.support_[i]]
        ranks    = dict(zip(features, rfe.ranking_.tolist()))
        result   = sorted(ranks.items(), key=lambda x: x[1])
        return jsonify({"method":"RFE (Logistic Regression)","selected": selected,
                        "result": [{"feature":k,"score":v} for k,v in result]})

    elif method == "variance":
        vt      = VarianceThreshold(threshold=0.01)
        vt.fit(X)
        kept    = [features[i] for i in range(len(features)) if vt.get_support()[i]]
        dropped = [features[i] for i in range(len(features)) if not vt.get_support()[i]]
        variances = dict(zip(features, [round(float(v),4) for v in X.var().values]))
        result  = sorted(variances.items(), key=lambda x: x[1], reverse=True)
        return jsonify({"method":"Variance Threshold","kept": kept, "dropped": dropped,
                        "result": [{"feature":k,"score":v} for k,v in result]})

    return jsonify({"error": "unknown method"})


if __name__ == "__main__":
    app.run(debug=True, port=5010)
