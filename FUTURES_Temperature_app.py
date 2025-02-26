import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
import graphviz

from io import BytesIO


def to_excel(df):
    """
    Convert a DataFrame to Excel bytes, suitable for a Streamlit download button.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='TrendAnalysis')
    return output.getvalue()


st.title("Temperature Analysis: Historical Trends & Future Predictions")

# Create two tabs
tab1, tab2 = st.tabs(["Trend Analysis", "Future Predictions"])

# ----------------------------------------------------------------------------
# TAB 1: TREND ANALYSIS
# ----------------------------------------------------------------------------
with tab1:
    st.header("1. Multi-Segment Straight-Line Trends & Anomalies")

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file (Temperature in °F) for Trend Analysis:",
        type=["csv", "xlsx"],
        key="tab1_uploader"
    )

    if uploaded_file is not None:
        # ---------------------------------------------------------------------
        # Read and clean the data
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df.dropna(inplace=True)

        # Rename 'Date' -> 'Year' if needed
        if "Date" in df.columns:
            df.rename(columns={"Date": "Year"}, inplace=True)

        # Check for 'Year' and 'Temperature'
        if "Year" not in df.columns:
            st.error("No 'Year' column found!")
            st.stop()
        if "Temperature" not in df.columns:
            st.error("No 'Temperature' column found! Please rename your column to 'Temperature'.")
            st.stop()

        # Convert 'Year' to numeric if needed
        if not pd.api.types.is_numeric_dtype(df["Year"]):
            df["Year"] = pd.to_datetime(df["Year"]).dt.year

        # Rename the Temperature column
        df.rename(columns={"Temperature": "TempF"}, inplace=True)

        # ---------------------------------------------------------------------
        # Option to display in °C or °F
        st.subheader("Temperature Units for Display")
        temp_unit = st.radio("Select unit:", ["Fahrenheit (°F)", "Celsius (°C)"])
        if temp_unit.startswith("Celsius"):
            df["TempDisplay"] = (df["TempF"] - 32.0) * 5.0/9.0
        else:
            df["TempDisplay"] = df["TempF"]

        st.write("### Preview of Uploaded Data (Cleaned)")
        st.dataframe(df.head())

        # ---------------------------------------------------------------------
        # Let the user define multiple straight-line trend segments
        st.subheader("Define Trend Segments")
        st.write(
            "Each segment will get its own baseline (mean of that segment) "
            "and a straight-line fit. Anomaly is computed as Temp / segment_mean."
        )

        years_sorted = sorted(df["Year"].unique())
        num_segments = st.number_input("How many trend segments?", min_value=0, max_value=10, value=1, step=1)

        segments = []
        for i in range(num_segments):
            st.markdown(f"**Segment #{i+1}**")
            col1, col2 = st.columns(2)
            with col1:
                seg_start = st.selectbox(f"Start Year (Segment {i+1})", years_sorted, key=f"start_{i}")
            with col2:
                seg_end = st.selectbox(f"End Year (Segment {i+1})", years_sorted, key=f"end_{i}")

            if seg_start <= seg_end:
                segments.append((seg_start, seg_end, i))
            else:
                st.warning(f"Segment {i+1}: Start year is greater than end year. Skipping...")

        # We'll build data needed for altair overlay
        lines_dfs = []
        anomalies_list = []

        # For each segment, compute baseline, anomaly, and linear regression
        for (start_yr, end_yr, idx) in segments:
            seg_data = df[(df["Year"] >= start_yr) & (df["Year"] <= end_yr)].copy()
            if seg_data.empty:
                continue

            seg_mean = seg_data["TempDisplay"].mean()
            seg_data[f"Anomaly_{idx}"] = seg_data["TempDisplay"] / seg_mean

            # Fit a straight line
            X = seg_data["Year"].values.reshape(-1, 1)
            y = seg_data["TempDisplay"].values
            linreg = LinearRegression().fit(X, y)
            seg_data[f"TrendPred_{idx}"] = linreg.predict(X)

            # Merge back
            df = df.merge(
                seg_data[["Year", f"Anomaly_{idx}", f"TrendPred_{idx}"]],
                on="Year", how="left"
            )

            # Create a small DataFrame with just the two endpoints for altair overlay
            m = linreg.coef_[0]
            b = linreg.intercept_
            seg_line_df = pd.DataFrame({
                "Year": [start_yr, end_yr],
                "y_line": [m*start_yr + b, m*end_yr + b],
                "SegmentIndex": [idx, idx]
            })
            lines_dfs.append(seg_line_df)

            # For plotting anomalies as points: 
            # We'll keep them in a "long" format with columns = ["Year", "Anomaly", "SegmentIndex"]
            sub_anomaly = seg_data[["Year", f"Anomaly_{idx}"]].rename(columns={f"Anomaly_{idx}": "Anomaly"}).copy()
            sub_anomaly["SegmentIndex"] = idx
            anomalies_list.append(sub_anomaly)

        # Combine line segments
        if lines_dfs:
            lines_combined = pd.concat(lines_dfs, ignore_index=True)
        else:
            lines_combined = pd.DataFrame(columns=["Year", "y_line", "SegmentIndex"])

        # Combine anomalies
        if anomalies_list:
            anomalies_combined = pd.concat(anomalies_list, ignore_index=True)
        else:
            anomalies_combined = pd.DataFrame(columns=["Year", "Anomaly", "SegmentIndex"])

        # ---------------------------------------------------------------------
        # Plot original data + each segment line in a different color + anomalies as points
        st.subheader("Historical Temperature with Straight-Line Trends & Anomalies")

        # Base chart for the actual temperature
        base_temp = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("TempDisplay:Q", title=f"Temp (°{'C' if temp_unit.startswith('Celsius') else 'F'})"),
            color=alt.value("blue"),  # original data in blue
            tooltip=["Year", "TempDisplay"]
        )

        # Overlay each segment line in a different color by SegmentIndex
        line_segments_chart = alt.Chart(lines_combined).mark_line(strokeWidth=3).encode(
            x=alt.X("Year:O"),
            y=alt.Y("y_line:Q"),
            color=alt.Color("SegmentIndex:N", title="Trend Segment"),
            tooltip=["SegmentIndex"]
        )

        # Plot anomalies as points with a separate Y axis (so it doesn't shift the main lines),
        # and color them by SegmentIndex. We'll do .resolve_scale(y='independent') later.
        anomaly_chart = alt.Chart(anomalies_combined).mark_point(size=60).encode(
            x=alt.X("Year:O"),
            y=alt.Y("Anomaly:Q", title="Anomaly (T / SegmentMean)"),
            color=alt.Color("SegmentIndex:N", title="Trend Segment"),
            tooltip=["Year", "Anomaly", "SegmentIndex"]
        )

        combined_chart = alt.layer(
            base_temp, 
            line_segments_chart, 
            anomaly_chart
        ).resolve_scale(y='independent').interactive()

        st.altair_chart(combined_chart, use_container_width=True)

        # ---------------------------------------------------------------------
        # Download the updated DataFrame with anomalies
        st.subheader("Download Augmented Data")
        excel_data = to_excel(df)
        st.download_button(
            label="Download data as Excel",
            data=excel_data,
            file_name="trend_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("Please upload a CSV or Excel file to define trends and view anomalies.")


# ----------------------------------------------------------------------------
# TAB 2: FUTURE PREDICTIONS (RANDOM FOREST)
# ----------------------------------------------------------------------------
with tab2:
    st.header("2. Future Predictions (Random Forest)")

    # File uploader for future predictions
    uploaded_file_2 = st.file_uploader(
        "Upload a CSV or Excel file (Temperature in °F) for Predictions:",
        type=["csv", "xlsx"],
        key="tab2_uploader"
    )

    if uploaded_file_2 is not None:
        file_ext = uploaded_file_2.name.split('.')[-1].lower()
        if file_ext == "csv":
            df2 = pd.read_csv(uploaded_file_2)
        else:
            df2 = pd.read_excel(uploaded_file_2)

        df2.dropna(inplace=True)

        if "Date" in df2.columns:
            df2.rename(columns={"Date": "Year"}, inplace=True)

        if "Year" not in df2.columns:
            st.error("No 'Year' column found!")
            st.stop()
        if "Temperature" not in df2.columns:
            st.error("No 'Temperature' column found! Please rename your column to 'Temperature'.")
            st.stop()

        if not pd.api.types.is_numeric_dtype(df2["Year"]):
            df2["Year"] = pd.to_datetime(df2["Year"]).dt.year

        df2.rename(columns={"Temperature": "TempF"}, inplace=True)

        st.write("### Preview of Historical Data")
        st.dataframe(df2.head())

        # Random Forest hyperparameter sliders
        st.subheader("Random Forest Model Hyperparameters")
        n_estimators = st.slider("Number of Trees", 10, 500, 100, step=10)
        max_depth = st.slider("Max Depth", 1, 30, 5)

        # Prepare X, y
        X = df2["Year"].values.reshape(-1, 1)
        y = df2["TempF"].values  # keep in °F

        # Fit the model
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        rf.fit(X, y)

        # Accuracy (R^2) in percentage
        r2 = rf.score(X, y)
        accuracy_percent = r2 * 100.0

        # Error (RMSE)
        preds_on_train = rf.predict(X)
        mse = mean_squared_error(y, preds_on_train)
        rmse = np.sqrt(mse)


        st.write(f"**Model Accuracy (R²)**: {r2:.3f}  ->  **{accuracy_percent:.2f}%**")
        st.write(f"**Root Mean Squared Error (°F)**: {rmse:.3f}")

        # Let user pick how many future years
        future_years = st.slider("Years to predict into the future", 1, 100, 20)

        last_year = df2["Year"].max()
        future_year_array = np.arange(last_year + 1, last_year + 1 + future_years)
        X_future = future_year_array.reshape(-1, 1)
        future_preds = rf.predict(X_future)

        future_df = pd.DataFrame({
            "Year": future_year_array,
            "Predicted_TempF": future_preds
        })

        st.write(f"### Next {future_years} Years of Predictions (°F)")
        st.dataframe(future_df.head(10))

        # Combine historical + future for plotting
        combined_df = pd.concat([
            df2.assign(Source="Historical").rename(columns={"TempF": "Temperature"}),
            future_df.assign(Source="Future").rename(columns={"Predicted_TempF": "Temperature"})
        ], ignore_index=True)

        st.write("### Historical + Future Predictions")
        chart_base = alt.Chart(combined_df).encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("Temperature:Q", title="Temperature (°F)"),
            color=alt.Color("Source:N", scale=alt.Scale(range=["blue", "orange"]))
        ).mark_line(point=True).encode(
            tooltip=["Year", "Temperature", "Source"]
        ).interactive()

        st.altair_chart(chart_base, use_container_width=True)

        # Optionally display the first tree structure
        st.subheader("Optional: Display the Structure of the First Tree")
        if st.button("Show Tree Structure"):
            # Export the first tree
            tree_dot = export_graphviz(
                rf.estimators_[0],
                out_file=None,
                feature_names=["Year"],
                filled=True,
                rounded=True
            )
            st.graphviz_chart(tree_dot)

    else:
        st.info("Please upload a CSV or Excel file for future predictions.")
