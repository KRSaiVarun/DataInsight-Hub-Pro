import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import re
from datetime import datetime, date
import xlsxwriter


# ─────────────────────────────────────────────
#  Formula Engine
# ─────────────────────────────────────────────

class FormulaEngine:
    """Parse and evaluate Excel-like formulas on a DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _col_values(self, col_name: str) -> pd.Series:
        col_name = col_name.strip().strip('"').strip("'")
        if col_name in self.df.columns:
            return pd.to_numeric(self.df[col_name], errors='coerce')
        raise ValueError(f"Column '{col_name}' not found")

    def _parse_args(self, args_str: str):
        """Split top-level comma-separated arguments."""
        args, depth, cur = [], 0, ""
        for ch in args_str:
            if ch == '(':
                depth += 1
                cur += ch
            elif ch == ')':
                depth -= 1
                cur += ch
            elif ch == ',' and depth == 0:
                args.append(cur.strip())
                cur = ""
            else:
                cur += ch
        if cur.strip():
            args.append(cur.strip())
        return args

    def _eval_arg(self, arg: str):
        """Evaluate a single argument — could be a number, string, col ref, or nested formula."""
        arg = arg.strip()
        # Nested formula
        if re.match(r'^[A-Z]+\(', arg):
            return self.evaluate(arg)
        # Quoted string
        if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
            return arg[1:-1]
        # Number
        try:
            return float(arg)
        except ValueError:
            pass
        # Boolean
        if arg.upper() == 'TRUE':
            return True
        if arg.upper() == 'FALSE':
            return False
        # Column reference
        if arg in self.df.columns:
            return self._col_values(arg)
        return arg

    def evaluate(self, formula: str):
        """Evaluate a formula string and return scalar or Series."""
        formula = formula.strip()
        if not formula.startswith('='):
            try:
                return float(formula)
            except ValueError:
                return formula

        formula = formula[1:].strip()
        m = re.match(r'^([A-Z]+)\((.*)\)$', formula, re.DOTALL)
        if not m:
            try:
                return float(formula)
            except ValueError:
                return formula

        func_name = m.group(1)
        args_str = m.group(2)
        args = self._parse_args(args_str)

        return self._dispatch(func_name, args)

    def _dispatch(self, func: str, args):
        evaled = [self._eval_arg(a) for a in args]

        def num_series(v):
            if isinstance(v, pd.Series):
                return v.dropna()
            return pd.Series([float(v)])

        # ── Math ──────────────────────────────
        if func == 'SUM':
            return sum(num_series(v).sum() for v in evaled)
        if func == 'AVERAGE':
            vals = pd.concat([num_series(v) for v in evaled])
            return vals.mean()
        if func == 'MIN':
            vals = pd.concat([num_series(v) for v in evaled])
            return vals.min()
        if func == 'MAX':
            vals = pd.concat([num_series(v) for v in evaled])
            return vals.max()
        if func == 'COUNT':
            return sum(len(num_series(v)) for v in evaled)
        if func == 'ROUND':
            val = float(num_series(evaled[0]).iloc[0]) if isinstance(evaled[0], pd.Series) else float(evaled[0])
            decimals = int(evaled[1]) if len(evaled) > 1 else 0
            return round(val, decimals)
        if func == 'ABS':
            val = float(num_series(evaled[0]).iloc[0]) if isinstance(evaled[0], pd.Series) else float(evaled[0])
            return abs(val)
        if func == 'POWER':
            base = float(evaled[0])
            exp = float(evaled[1])
            return base ** exp
        if func == 'SQRT':
            val = float(evaled[0])
            return val ** 0.5

        # ── Logical ───────────────────────────
        if func == 'IF':
            cond, true_val, false_val = evaled[0], evaled[1], evaled[2] if len(evaled) > 2 else None
            if isinstance(cond, pd.Series):
                return cond.map(lambda x: true_val if x else false_val)
            return true_val if cond else false_val
        if func == 'AND':
            return all(bool(v) for v in evaled)
        if func == 'OR':
            return any(bool(v) for v in evaled)
        if func == 'NOT':
            return not bool(evaled[0])

        # ── Text ──────────────────────────────
        if func == 'CONCAT':
            return ''.join(str(v) for v in evaled)
        if func == 'LEFT':
            text = str(evaled[0])
            n = int(evaled[1]) if len(evaled) > 1 else 1
            return text[:n]
        if func == 'RIGHT':
            text = str(evaled[0])
            n = int(evaled[1]) if len(evaled) > 1 else 1
            return text[-n:]
        if func == 'MID':
            text = str(evaled[0])
            start = int(evaled[1]) - 1
            n = int(evaled[2])
            return text[start:start + n]
        if func == 'LEN':
            return len(str(evaled[0]))
        if func == 'UPPER':
            return str(evaled[0]).upper()
        if func == 'LOWER':
            return str(evaled[0]).lower()
        if func == 'TRIM':
            return str(evaled[0]).strip()

        # ── Lookup ────────────────────────────
        if func == 'VLOOKUP':
            lookup_val, lookup_col, return_col = evaled[0], str(evaled[1]), str(evaled[2])
            match = self.df[self.df[lookup_col].astype(str) == str(lookup_val)]
            if not match.empty:
                return match.iloc[0][return_col]
            return '#N/A'
        if func == 'MATCH':
            lookup_val, lookup_col = evaled[0], str(evaled[1])
            matches = self.df[self.df[lookup_col].astype(str) == str(lookup_val)].index.tolist()
            return matches[0] + 1 if matches else '#N/A'
        if func == 'INDEX':
            col = str(evaled[0])
            row = int(evaled[1]) - 1
            if col in self.df.columns and 0 <= row < len(self.df):
                return self.df.iloc[row][col]
            return '#REF!'

        # ── Date ──────────────────────────────
        if func == 'TODAY':
            return str(date.today())
        if func == 'NOW':
            return str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        if func == 'YEAR':
            return pd.to_datetime(str(evaled[0])).year
        if func == 'MONTH':
            return pd.to_datetime(str(evaled[0])).month
        if func == 'DAY':
            return pd.to_datetime(str(evaled[0])).day
        if func == 'DATEDIF':
            d1 = pd.to_datetime(str(evaled[0]))
            d2 = pd.to_datetime(str(evaled[1]))
            unit = str(evaled[2]).upper() if len(evaled) > 2 else 'D'
            delta = d2 - d1
            if unit == 'D':
                return delta.days
            if unit == 'M':
                return (d2.year - d1.year) * 12 + d2.month - d1.month
            if unit == 'Y':
                return d2.year - d1.year

        return f'#NAME? ({func})'


# ─────────────────────────────────────────────
#  AI Assistant (rule-based)
# ─────────────────────────────────────────────

class ExcelAIAssistant:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def process(self, query: str):
        q = query.lower().strip()
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Growth / trend
        if 'growth' in q or 'trend' in q:
            if num_cols:
                col = num_cols[0]
                for nc in num_cols:
                    if any(w in nc.lower() for w in ['sales', 'revenue', 'amount', 'profit']):
                        col = nc
                        break
                result_df = self.df[[col]].copy()
                result_df['Running Total'] = result_df[col].cumsum()
                result_df['Period Growth %'] = result_df[col].pct_change() * 100
                return "📈 Trend Analysis", result_df, None

        # Duplicates
        if 'duplicate' in q:
            dups = self.df[self.df.duplicated(keep=False)]
            return "🔍 Duplicate Rows", dups, f"Found {len(dups)} duplicate rows"

        # Pivot / group by department / category
        if 'pivot' in q or 'group' in q or 'department' in q or 'category' in q:
            if cat_cols and num_cols:
                grp_col = cat_cols[0]
                for cc in cat_cols:
                    if any(w in cc.lower() for w in ['dept', 'department', 'category', 'region', 'type']):
                        grp_col = cc
                        break
                val_col = num_cols[0]
                pivot = self.df.groupby(grp_col)[val_col].agg(['sum', 'mean', 'count']).reset_index()
                pivot.columns = [grp_col, 'Total', 'Average', 'Count']
                fig = px.bar(pivot, x=grp_col, y='Total', title=f'Total {val_col} by {grp_col}', color=grp_col)
                return f"📊 Pivot by {grp_col}", pivot, fig

        # Top customers / records
        if 'top' in q:
            n = 10
            m = re.search(r'top\s+(\d+)', q)
            if m:
                n = int(m.group(1))
            if num_cols:
                val_col = num_cols[0]
                for nc in num_cols:
                    if any(w in nc.lower() for w in ['revenue', 'sales', 'amount', 'profit', 'score']):
                        val_col = nc
                        break
                top_df = self.df.nlargest(n, val_col)
                return f"🏆 Top {n} by {val_col}", top_df, None

        # Missing values
        if 'missing' in q or 'null' in q or 'nan' in q:
            missing = self.df.isnull().sum().reset_index()
            missing.columns = ['Column', 'Missing Count']
            missing['Missing %'] = (missing['Missing Count'] / len(self.df) * 100).round(2)
            missing = missing[missing['Missing Count'] > 0]
            return "❓ Missing Values", missing, None

        # Correlation
        if 'correlat' in q:
            if len(num_cols) >= 2:
                corr = self.df[num_cols].corr().round(3)
                fig = px.imshow(corr, text_auto=True, title='Correlation Matrix', color_continuous_scale='RdBu_r')
                return "🔗 Correlation Analysis", corr, fig

        # Summary stats
        if 'summary' in q or 'statistic' in q or 'describe' in q:
            desc = self.df.describe().round(3)
            return "📋 Summary Statistics", desc, None

        # Profit margin / margin
        if 'margin' in q or 'profit' in q:
            profit_col = next((c for c in num_cols if 'profit' in c.lower()), None)
            revenue_col = next((c for c in num_cols if 'revenue' in c.lower() or 'sales' in c.lower()), None)
            if profit_col and revenue_col:
                result_df = self.df[[profit_col, revenue_col]].copy()
                result_df['Profit Margin %'] = (result_df[profit_col] / result_df[revenue_col] * 100).round(2)
                return "💰 Profit Margin", result_df, None

        # Default: show column summaries
        summary_rows = []
        for col in self.df.columns:
            row = {'Column': col, 'Type': str(self.df[col].dtype), 'Non-Null': self.df[col].count(), 'Unique': self.df[col].nunique()}
            if self.df[col].dtype in [np.float64, np.int64]:
                row['Min'] = self.df[col].min()
                row['Max'] = self.df[col].max()
                row['Mean'] = round(self.df[col].mean(), 2)
            summary_rows.append(row)
        return "📊 Dataset Overview", pd.DataFrame(summary_rows), None


# ─────────────────────────────────────────────
#  Main ExcelWorkspace
# ─────────────────────────────────────────────

class ExcelWorkspace:
    """Advanced Excel-like workspace for DataInsightHub."""

    def _init_session(self):
        defaults = {
            'ew_sheets': {'Sheet1': pd.DataFrame()},
            'ew_active': 'Sheet1',
            'ew_undo_stack': [],
            'ew_redo_stack': [],
            'ew_formula_results': {},
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    # ── Sheet helpers ──────────────────────────

    def _active_df(self) -> pd.DataFrame:
        return st.session_state.ew_sheets[st.session_state.ew_active]

    def _set_active_df(self, df: pd.DataFrame, push_undo=True):
        if push_undo:
            st.session_state.ew_undo_stack.append(
                (st.session_state.ew_active, self._active_df().copy())
            )
            st.session_state.ew_redo_stack.clear()
        st.session_state.ew_sheets[st.session_state.ew_active] = df

    # ── Export helpers ─────────────────────────

    def _to_xlsx_bytes(self, df: pd.DataFrame) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name=st.session_state.ew_active)
            wb = writer.book
            ws = writer.sheets[st.session_state.ew_active]
            header_fmt = wb.add_format({'bold': True, 'bg_color': '#4B5563', 'font_color': 'white', 'border': 1})
            for col_num, col_name in enumerate(df.columns):
                ws.write(0, col_num, col_name, header_fmt)
                ws.set_column(col_num, col_num, max(len(str(col_name)) + 4, 12))
        return buf.getvalue()

    def _fig_to_png(self, fig) -> bytes:
        return fig.to_image(format='png', width=1200, height=700)

    # ── Render sections ───────────────────────

    def _render_sheet_tabs(self):
        sheets = list(st.session_state.ew_sheets.keys())
        col_tabs = st.columns(len(sheets) + 3)
        for i, name in enumerate(sheets):
            label = f"{'📄 ' if name == st.session_state.ew_active else ''}{name}"
            if col_tabs[i].button(label, key=f"sheet_tab_{name}"):
                st.session_state.ew_active = name
                st.rerun()

        if col_tabs[len(sheets)].button("➕ Add Sheet"):
            new_name = f"Sheet{len(sheets) + 1}"
            st.session_state.ew_sheets[new_name] = pd.DataFrame()
            st.session_state.ew_active = new_name
            st.rerun()

        if col_tabs[len(sheets) + 1].button("📋 Duplicate"):
            new_name = f"{st.session_state.ew_active}_copy"
            st.session_state.ew_sheets[new_name] = self._active_df().copy()
            st.session_state.ew_active = new_name
            st.rerun()

        if col_tabs[len(sheets) + 2].button("🗑️ Delete Sheet") and len(sheets) > 1:
            del st.session_state.ew_sheets[st.session_state.ew_active]
            st.session_state.ew_active = list(st.session_state.ew_sheets.keys())[0]
            st.rerun()

    def _render_toolbar(self, df: pd.DataFrame):
        """Toolbar: undo/redo, add/delete rows & cols, rename col."""
        st.markdown("##### 🛠️ Toolbar")
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

        with c1:
            if st.button("↩️ Undo", disabled=not st.session_state.ew_undo_stack):
                sheet, prev_df = st.session_state.ew_undo_stack.pop()
                st.session_state.ew_redo_stack.append((sheet, self._active_df().copy()))
                st.session_state.ew_sheets[sheet] = prev_df
                st.rerun()
        with c2:
            if st.button("↪️ Redo", disabled=not st.session_state.ew_redo_stack):
                sheet, next_df = st.session_state.ew_redo_stack.pop()
                st.session_state.ew_undo_stack.append((sheet, self._active_df().copy()))
                st.session_state.ew_sheets[sheet] = next_df
                st.rerun()
        with c3:
            if st.button("➕ Add Row"):
                new_row = pd.DataFrame([{c: None for c in df.columns}])
                self._set_active_df(pd.concat([df, new_row], ignore_index=True))
                st.rerun()
        with c4:
            if st.button("➖ Del Row") and len(df) > 0:
                self._set_active_df(df.iloc[:-1].reset_index(drop=True))
                st.rerun()
        with c5:
            new_col = st.text_input("New column name", label_visibility="collapsed", placeholder="New column name", key="ew_new_col")
        with c6:
            if st.button("➕ Add Col") and new_col:
                new_df = df.copy()
                new_df[new_col] = None
                self._set_active_df(new_df)
                st.rerun()
        with c7:
            if st.button("➖ Del Last Col") and len(df.columns) > 0:
                self._set_active_df(df.drop(columns=[df.columns[-1]]))
                st.rerun()

        # Rename column
        with st.expander("✏️ Rename Column"):
            rc1, rc2, rc3 = st.columns(3)
            old_name = rc1.selectbox("Column to rename", df.columns.tolist(), key="ew_rename_old")
            new_name = rc2.text_input("New name", key="ew_rename_new")
            if rc3.button("Rename") and new_name and old_name:
                new_df = df.rename(columns={old_name: new_name})
                self._set_active_df(new_df)
                st.rerun()

    def _render_formula_bar(self, df: pd.DataFrame):
        st.markdown("##### ƒ Formula Bar")
        fc1, fc2, fc3 = st.columns([2, 5, 1])
        new_col_name = fc1.text_input("Result column name", value="Formula_Result", key="ew_fcol")
        formula_input = fc2.text_input("Enter formula (e.g. =SUM(Sales))", key="ew_formula")
        run_formula = fc3.button("▶ Run", key="ew_run_formula")

        if run_formula and formula_input:
            try:
                engine = FormulaEngine(df)
                result = engine.evaluate(formula_input)
                if isinstance(result, pd.Series):
                    new_df = df.copy()
                    new_df[new_col_name] = result.values[:len(df)] if len(result) >= len(df) else list(result) + [None] * (len(df) - len(result))
                    self._set_active_df(new_df)
                    st.success(f"✅ Column '{new_col_name}' added")
                else:
                    st.session_state.ew_formula_results[new_col_name] = result
                    st.success(f"✅ Result: **{result}**")
            except Exception as e:
                st.error(f"Formula error: {e}")

        if st.session_state.ew_formula_results:
            with st.expander("📋 Scalar Results"):
                for k, v in st.session_state.ew_formula_results.items():
                    st.markdown(f"**{k}** = `{v}`")

        with st.expander("📖 Formula Reference"):
            st.markdown("""
**Math:** `=SUM(Col)` `=AVERAGE(Col)` `=MIN(Col)` `=MAX(Col)` `=COUNT(Col)` `=ROUND(Col,2)` `=ABS(Col)` `=POWER(2,10)` `=SQRT(Col)`

**Logical:** `=IF(Col,Yes,No)` `=AND(...)` `=OR(...)` `=NOT(...)`

**Text:** `=CONCAT(Col1,Col2)` `=UPPER(Col)` `=LOWER(Col)` `=TRIM(Col)` `=LEN(Col)` `=LEFT(Col,3)` `=RIGHT(Col,3)` `=MID(Col,2,4)`

**Lookup:** `=VLOOKUP(value,LookupCol,ReturnCol)` `=MATCH(value,Col)` `=INDEX(Col,row)`

**Date:** `=TODAY()` `=NOW()` `=YEAR(DateCol)` `=MONTH(DateCol)` `=DAY(DateCol)` `=DATEDIF(Date1,Date2,D)`
""")

    def _render_data_grid(self, df: pd.DataFrame):
        st.markdown("##### 📋 Spreadsheet")
        if df.empty:
            st.info("Sheet is empty. Add columns using the toolbar above or import data from the sidebar.")
            return

        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            height=420,
            key=f"ew_grid_{st.session_state.ew_active}",
        )
        if not edited.equals(df):
            self._set_active_df(edited)

    def _render_data_cleaning(self, df: pd.DataFrame):
        st.markdown("##### 🧹 Data Cleaning Tools")
        if df.empty:
            st.info("No data to clean.")
            return

        cols = st.columns(4)

        with cols[0]:
            if st.button("🔁 Remove Duplicates"):
                before = len(df)
                new_df = df.drop_duplicates().reset_index(drop=True)
                self._set_active_df(new_df)
                st.success(f"Removed {before - len(new_df)} duplicate rows")
                st.rerun()

        with cols[1]:
            fill_method = st.selectbox("Fill missing with:", ["mean", "median", "mode", "0", "forward fill", "backward fill"], key="ew_fill_method")
            if st.button("🔧 Fill Missing"):
                new_df = df.copy()
                for col in new_df.columns:
                    if new_df[col].isnull().any():
                        if fill_method == "mean" and pd.api.types.is_numeric_dtype(new_df[col]):
                            new_df[col].fillna(new_df[col].mean(), inplace=True)
                        elif fill_method == "median" and pd.api.types.is_numeric_dtype(new_df[col]):
                            new_df[col].fillna(new_df[col].median(), inplace=True)
                        elif fill_method == "mode":
                            new_df[col].fillna(new_df[col].mode()[0] if not new_df[col].mode().empty else "", inplace=True)
                        elif fill_method == "0":
                            new_df[col].fillna(0, inplace=True)
                        elif fill_method == "forward fill":
                            new_df[col].fillna(method='ffill', inplace=True)
                        elif fill_method == "backward fill":
                            new_df[col].fillna(method='bfill', inplace=True)
                self._set_active_df(new_df)
                st.success("Missing values filled")
                st.rerun()

        with cols[2]:
            if st.button("✂️ Trim Spaces"):
                new_df = df.copy()
                for col in new_df.select_dtypes(include='object').columns:
                    new_df[col] = new_df[col].str.strip()
                self._set_active_df(new_df)
                st.success("Spaces trimmed")
                st.rerun()

        with cols[3]:
            txt_col = st.selectbox("Column", df.columns.tolist(), key="ew_txt2num_col")
            if st.button("🔢 Text → Number"):
                new_df = df.copy()
                new_df[txt_col] = pd.to_numeric(new_df[txt_col], errors='coerce')
                self._set_active_df(new_df)
                st.success(f"'{txt_col}' converted to numeric")
                st.rerun()

        # Replace values
        with st.expander("🔄 Replace Values"):
            rc1, rc2, rc3, rc4 = st.columns(4)
            rep_col = rc1.selectbox("Column", ["All columns"] + df.columns.tolist(), key="ew_rep_col")
            find_val = rc2.text_input("Find", key="ew_find_val")
            replace_val = rc3.text_input("Replace with", key="ew_replace_val")
            if rc4.button("Replace", key="ew_do_replace"):
                new_df = df.copy()
                if rep_col == "All columns":
                    new_df = new_df.replace(find_val, replace_val)
                else:
                    new_df[rep_col] = new_df[rep_col].replace(find_val, replace_val)
                self._set_active_df(new_df)
                st.success("Values replaced")
                st.rerun()

        # Split / Merge columns
        with st.expander("✂️ Split & Merge Columns"):
            s1, s2, s3, s4 = st.columns(4)
            split_col = s1.selectbox("Split column", df.select_dtypes(include='object').columns.tolist() or df.columns.tolist(), key="ew_split_col")
            delimiter = s2.text_input("Delimiter", value=",", key="ew_delim")
            if s3.button("Split Column"):
                new_df = df.copy()
                split_result = new_df[split_col].str.split(delimiter, expand=True)
                for i, c in enumerate(split_result.columns):
                    new_df[f"{split_col}_part{i+1}"] = split_result[c]
                self._set_active_df(new_df)
                st.rerun()

            m1, m2, m3, m4 = st.columns(4)
            merge_cols = m1.multiselect("Columns to merge", df.columns.tolist(), key="ew_merge_cols")
            merge_sep = m2.text_input("Separator", value=" ", key="ew_merge_sep")
            merge_name = m3.text_input("New column name", value="Merged", key="ew_merge_name")
            if m4.button("Merge Columns") and merge_cols:
                new_df = df.copy()
                new_df[merge_name] = new_df[merge_cols].astype(str).agg(merge_sep.join, axis=1)
                self._set_active_df(new_df)
                st.rerun()

        # Outlier detection
        with st.expander("📡 Find Outliers (IQR Method)"):
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                out_col = st.selectbox("Column", num_cols, key="ew_outlier_col")
                if st.button("Detect Outliers"):
                    Q1 = df[out_col].quantile(0.25)
                    Q3 = df[out_col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[out_col] < Q1 - 1.5 * IQR) | (df[out_col] > Q3 + 1.5 * IQR)]
                    st.write(f"Found **{len(outliers)}** outliers in `{out_col}`")
                    if not outliers.empty:
                        st.dataframe(outliers, use_container_width=True)

    def _render_analytics(self, df: pd.DataFrame):
        st.markdown("##### 📊 Analytics")
        if df.empty:
            st.info("No data for analytics.")
            return

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Pivot Table", "Group By", "Running Totals", "Ranking", "Correlation"])

        with tab1:
            if cat_cols and num_cols:
                pc1, pc2, pc3 = st.columns(3)
                row_col = pc1.selectbox("Row (Group By)", cat_cols, key="ew_piv_row")
                val_col = pc2.selectbox("Value", num_cols, key="ew_piv_val")
                agg_func = pc3.selectbox("Aggregate", ["sum", "mean", "count", "min", "max"], key="ew_piv_agg")
                pivot = df.groupby(row_col)[val_col].agg(agg_func).reset_index()
                pivot.columns = [row_col, f"{agg_func}({val_col})"]
                st.dataframe(pivot, use_container_width=True)
                fig = px.bar(pivot, x=row_col, y=f"{agg_func}({val_col})", title=f"{agg_func.title()} of {val_col} by {row_col}", color=row_col)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least one categorical and one numeric column.")

        with tab2:
            if cat_cols and num_cols:
                gc1, gc2 = st.columns(2)
                grp_cols = gc1.multiselect("Group by", cat_cols, default=cat_cols[:1], key="ew_grp_cols")
                agg_cols = gc2.multiselect("Aggregate columns", num_cols, default=num_cols[:min(3, len(num_cols))], key="ew_grp_agg_cols")
                if grp_cols and agg_cols:
                    grouped = df.groupby(grp_cols)[agg_cols].agg(['sum', 'mean', 'count']).round(2)
                    grouped.columns = [f"{c[0]}_{c[1]}" for c in grouped.columns]
                    grouped = grouped.reset_index()
                    st.dataframe(grouped, use_container_width=True)

        with tab3:
            if num_cols:
                rt_col = st.selectbox("Column for running total", num_cols, key="ew_rt_col")
                result = df.copy()
                result[f'{rt_col}_Running_Total'] = result[rt_col].cumsum()
                result[f'{rt_col}_Running_Avg'] = result[rt_col].expanding().mean().round(2)
                result[f'{rt_col}_Pct_Contribution'] = (result[rt_col] / result[rt_col].sum() * 100).round(2)
                display_cols = [rt_col, f'{rt_col}_Running_Total', f'{rt_col}_Running_Avg', f'{rt_col}_Pct_Contribution']
                st.dataframe(result[display_cols], use_container_width=True)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=result.index, y=result[rt_col], name='Values'))
                fig.add_trace(go.Scatter(x=result.index, y=result[f'{rt_col}_Running_Total'], name='Running Total', yaxis='y2'))
                fig.update_layout(
                    title=f'Running Total — {rt_col}',
                    yaxis2=dict(overlaying='y', side='right'),
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            if num_cols:
                rk_col = st.selectbox("Rank by", num_cols, key="ew_rank_col")
                rk_asc = st.checkbox("Ascending", value=False, key="ew_rank_asc")
                result = df.copy()
                result['Rank'] = result[rk_col].rank(ascending=rk_asc, method='dense').astype(int)
                result['Percentile'] = (result[rk_col].rank(pct=True) * 100).round(1)
                st.dataframe(result.sort_values('Rank'), use_container_width=True)

        with tab5:
            if len(num_cols) >= 2:
                corr = df[num_cols].corr().round(3)
                fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title='Correlation Matrix', aspect='auto')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(corr, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns.")

    def _render_conditional_formatting(self, df: pd.DataFrame):
        st.markdown("##### 🎨 Conditional Formatting Preview")
        if df.empty:
            st.info("No data to format.")
            return

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.info("No numeric columns for formatting.")
            return

        cf1, cf2 = st.columns(2)
        cf_col = cf1.selectbox("Column", num_cols, key="ew_cf_col")
        cf_type = cf2.selectbox("Formatting Type", [
            "Color Scale (Green→Red)", "Color Scale (Red→Green)",
            "Highlight Top 10%", "Highlight Bottom 10%",
            "Highlight Above Average", "Highlight Below Average",
            "Highlight Duplicates", "Highlight Blanks"
        ], key="ew_cf_type")

        if st.button("Apply Formatting Preview"):
            styled = df.copy()
            col_data = df[cf_col]

            def style_series(s):
                styles = [''] * len(s)
                if cf_type == "Color Scale (Green→Red)":
                    norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
                    return [f'background-color: rgba({int(255*v)},{int(255*(1-v))},0,0.5)' for v in norm]
                elif cf_type == "Color Scale (Red→Green)":
                    norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
                    return [f'background-color: rgba({int(255*(1-v))},{int(255*v)},0,0.5)' for v in norm]
                elif cf_type == "Highlight Top 10%":
                    threshold = s.quantile(0.9)
                    return ['background-color: #16a34a; color: white' if v >= threshold else '' for v in s]
                elif cf_type == "Highlight Bottom 10%":
                    threshold = s.quantile(0.1)
                    return ['background-color: #dc2626; color: white' if v <= threshold else '' for v in s]
                elif cf_type == "Highlight Above Average":
                    avg = s.mean()
                    return ['background-color: #2563eb; color: white' if v > avg else '' for v in s]
                elif cf_type == "Highlight Below Average":
                    avg = s.mean()
                    return ['background-color: #d97706; color: white' if v < avg else '' for v in s]
                return styles

            if cf_type in ["Highlight Duplicates"]:
                def style_df(df_inner):
                    result = pd.DataFrame('', index=df_inner.index, columns=df_inner.columns)
                    dup_mask = df_inner[cf_col].duplicated(keep=False)
                    result.loc[dup_mask, cf_col] = 'background-color: #7c3aed; color: white'
                    return result
                st.dataframe(styled.style.apply(style_df, axis=None), use_container_width=True, height=400)
            elif cf_type == "Highlight Blanks":
                def style_blanks(df_inner):
                    result = pd.DataFrame('', index=df_inner.index, columns=df_inner.columns)
                    blank_mask = df_inner[cf_col].isnull()
                    result.loc[blank_mask, cf_col] = 'background-color: #6b7280; color: white'
                    return result
                st.dataframe(styled.style.apply(style_blanks, axis=None), use_container_width=True, height=400)
            else:
                st.dataframe(styled.style.apply(style_series, subset=[cf_col]), use_container_width=True, height=400)

    def _render_sorting_filtering(self, df: pd.DataFrame):
        st.markdown("##### 🔍 Sort & Filter")
        if df.empty:
            st.info("No data.")
            return

        with st.expander("🔃 Multi-column Sort"):
            sort_cols = st.multiselect("Sort by columns", df.columns.tolist(), key="ew_sort_cols")
            asc_flags = [st.checkbox(f"Ascending: {c}", value=True, key=f"ew_sort_asc_{c}") for c in sort_cols]
            if st.button("Apply Sort") and sort_cols:
                self._set_active_df(df.sort_values(sort_cols, ascending=asc_flags).reset_index(drop=True))
                st.rerun()

        with st.expander("🔎 Search Within Columns"):
            sc1, sc2 = st.columns(2)
            search_col = sc1.selectbox("Column", df.columns.tolist(), key="ew_search_col")
            search_val = sc2.text_input("Search term", key="ew_search_val")
            if search_val:
                result = df[df[search_col].astype(str).str.contains(search_val, case=False, na=False)]
                st.write(f"Found **{len(result)}** matches")
                st.dataframe(result, use_container_width=True)

        with st.expander("🔢 Numeric Range Filter"):
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                nr_col = st.selectbox("Column", num_cols, key="ew_nr_col")
                nr_min, nr_max = float(df[nr_col].min()), float(df[nr_col].max())
                nr_range = st.slider("Range", nr_min, nr_max, (nr_min, nr_max), key="ew_nr_range")
                filtered = df[(df[nr_col] >= nr_range[0]) & (df[nr_col] <= nr_range[1])]
                st.write(f"**{len(filtered)}** rows match")
                st.dataframe(filtered, use_container_width=True)

        with st.expander("🗂️ Category Filter"):
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                cat_col = st.selectbox("Column", cat_cols, key="ew_cat_col")
                options = df[cat_col].dropna().unique().tolist()
                selected = st.multiselect("Values", options, default=options, key="ew_cat_sel")
                if selected:
                    filtered = df[df[cat_col].isin(selected)]
                    st.write(f"**{len(filtered)}** rows match")
                    st.dataframe(filtered, use_container_width=True)

        with st.expander("📅 Date Range Filter"):
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if date_cols:
                dc = st.selectbox("Date column", date_cols, key="ew_date_col")
                try:
                    df_dates = pd.to_datetime(df[dc], errors='coerce')
                    d_min = df_dates.min().date()
                    d_max = df_dates.max().date()
                    d_range = st.date_input("Date range", (d_min, d_max), key="ew_date_range")
                    if len(d_range) == 2:
                        mask = (df_dates.dt.date >= d_range[0]) & (df_dates.dt.date <= d_range[1])
                        st.dataframe(df[mask], use_container_width=True)
                except Exception:
                    st.info("Could not parse dates in selected column.")
            else:
                st.info("No date-like columns found.")

    def _render_charts(self, df: pd.DataFrame):
        st.markdown("##### 📈 Excel Charts")
        if df.empty:
            st.info("No data to chart.")
            return

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        ch1, ch2, ch3 = st.columns(3)
        chart_type = ch1.selectbox("Chart Type", [
            "Bar", "Line", "Pie", "Area", "Scatter",
            "Histogram", "Heatmap", "Box Plot", "Waterfall", "Funnel"
        ], key="ew_chart_type")

        fig = None

        if chart_type in ["Bar", "Line", "Area"]:
            x_col = ch2.selectbox("X axis", df.columns.tolist(), key="ew_x_col")
            y_col = ch3.selectbox("Y axis", num_cols, key="ew_y_col")
            color_col = st.selectbox("Color by (optional)", ["None"] + cat_cols, key="ew_color_col")
            color_col = None if color_col == "None" else color_col
            if chart_type == "Bar":
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}", template='plotly_dark')
            elif chart_type == "Line":
                fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} over {x_col}", template='plotly_dark')
            elif chart_type == "Area":
                fig = px.area(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} area over {x_col}", template='plotly_dark')

        elif chart_type == "Scatter":
            x_col = ch2.selectbox("X axis", num_cols, key="ew_scatter_x")
            y_col = ch3.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="ew_scatter_y")
            color_col = st.selectbox("Color by", ["None"] + cat_cols, key="ew_scatter_color")
            color_col = None if color_col == "None" else color_col
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}", template='plotly_dark')

        elif chart_type == "Pie":
            val_col = ch2.selectbox("Values", num_cols, key="ew_pie_val")
            name_col = ch3.selectbox("Labels", df.columns.tolist(), key="ew_pie_name")
            pie_data = df.groupby(name_col)[val_col].sum().reset_index()
            fig = px.pie(pie_data, values=val_col, names=name_col, title=f"{val_col} distribution", template='plotly_dark')

        elif chart_type == "Histogram":
            h_col = ch2.selectbox("Column", num_cols, key="ew_hist_col")
            bins = ch3.slider("Bins", 5, 100, 30, key="ew_hist_bins")
            fig = px.histogram(df, x=h_col, nbins=bins, title=f"Distribution of {h_col}", template='plotly_dark')

        elif chart_type == "Heatmap":
            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title='Correlation Heatmap', template='plotly_dark')

        elif chart_type == "Box Plot":
            bp_col = ch2.selectbox("Column", num_cols, key="ew_bp_col")
            grp_col = ch3.selectbox("Group by", ["None"] + cat_cols, key="ew_bp_grp")
            grp_col = None if grp_col == "None" else grp_col
            fig = px.box(df, x=grp_col, y=bp_col, title=f"Box Plot — {bp_col}", template='plotly_dark')

        elif chart_type == "Waterfall":
            if num_cols and cat_cols:
                wf_val = ch2.selectbox("Values", num_cols, key="ew_wf_val")
                wf_label = ch3.selectbox("Labels", cat_cols, key="ew_wf_label")
                wf_data = df.groupby(wf_label)[wf_val].sum().reset_index()
                fig = go.Figure(go.Waterfall(
                    name="Waterfall", orientation="v",
                    x=wf_data[wf_label].tolist(),
                    y=wf_data[wf_val].tolist(),
                    connector={"line": {"color": "rgb(63,63,63)"}},
                ))
                fig.update_layout(title=f"Waterfall — {wf_val}", template='plotly_dark')

        elif chart_type == "Funnel":
            if num_cols and cat_cols:
                fn_val = ch2.selectbox("Values", num_cols, key="ew_fn_val")
                fn_label = ch3.selectbox("Labels", cat_cols, key="ew_fn_label")
                fn_data = df.groupby(fn_label)[fn_val].sum().reset_index().sort_values(fn_val, ascending=False)
                fig = px.funnel(fn_data, x=fn_val, y=fn_label, title=f"Funnel — {fn_val}", template='plotly_dark')

        if fig:
            st.plotly_chart(fig, use_container_width=True)
            try:
                png_bytes = self._fig_to_png(fig)
                st.download_button("📥 Download Chart (PNG)", data=png_bytes, file_name=f"chart_{chart_type.lower()}.png", mime="image/png")
            except Exception:
                pass

    def _render_ai_assistant(self, df: pd.DataFrame):
        st.markdown("##### 🤖 AI Excel Assistant")
        st.caption("Ask questions in plain English about your data")

        examples = [
            "Calculate monthly sales growth",
            "Create a pivot table by department",
            "Find duplicate employee IDs",
            "Show top 10 customers by revenue",
            "Generate a profit margin chart",
            "Show correlation analysis",
            "Find missing values",
            "Show summary statistics",
        ]

        ex_choice = st.selectbox("💡 Example queries:", ["(select or type below)"] + examples, key="ew_ai_example")
        query = st.text_input("Your query:", value=ex_choice if ex_choice != "(select or type below)" else "", key="ew_ai_query")

        if st.button("▶ Run Query", key="ew_ai_run") and query and not df.empty:
            with st.spinner("Analyzing..."):
                try:
                    assistant = ExcelAIAssistant(df)
                    title, result_df, chart = assistant.process(query)
                    st.subheader(title)
                    if result_df is not None and not result_df.empty:
                        st.dataframe(result_df, use_container_width=True)
                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
        elif df.empty:
            st.info("Load data first to use the AI assistant.")

    def _render_export(self, df: pd.DataFrame):
        st.markdown("##### 💾 Export")
        if df.empty:
            st.info("No data to export.")
            return

        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            xlsx_bytes = self._to_xlsx_bytes(df)
            st.download_button(
                "📥 Download XLSX",
                data=xlsx_bytes,
                file_name=f"{st.session_state.ew_active}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with ex2:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name=f"{st.session_state.ew_active}.csv",
                mime="text/csv"
            )
        with ex3:
            st.caption("Export all sheets to XLSX")
            if st.button("📦 Export All Sheets"):
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                    for sheet_name, sheet_df in st.session_state.ew_sheets.items():
                        if not sheet_df.empty:
                            sheet_df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
                st.download_button(
                    "📥 Download Multi-Sheet XLSX",
                    data=buf.getvalue(),
                    file_name="workbook_all_sheets.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # ── Main render ───────────────────────────

    def render(self, uploaded_df: pd.DataFrame | None = None):
        self._init_session()

        # Sync data from sidebar upload into active sheet if provided
        if uploaded_df is not None:
            active = st.session_state.ew_active
            if st.session_state.ew_sheets.get(active, pd.DataFrame()).empty:
                st.session_state.ew_sheets[active] = uploaded_df.copy()

        st.markdown("""
        <style>
        .excel-header {
            background: linear-gradient(135deg, #1e3a5f 0%, #0d6efd 100%);
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 16px;
        }
        </style>
        <div class="excel-header">
            <h2 style="color:white;margin:0;">📊 Excel Workspace</h2>
            <p style="color:#a8c7fa;margin:0;font-size:13px;">Full-featured spreadsheet environment with formulas, analytics & AI assistant</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Sheet tabs ────────────────────────
        self._render_sheet_tabs()
        st.markdown("---")

        df = self._active_df()

        # ── Import from main data ─────────────
        if uploaded_df is not None and not uploaded_df.empty:
            col_imp1, col_imp2 = st.columns([3, 1])
            col_imp1.caption(f"📂 Main dataset available ({len(uploaded_df):,} rows × {len(uploaded_df.columns)} cols)")
            if col_imp2.button("⬇ Import to Active Sheet"):
                self._set_active_df(uploaded_df.copy())
                st.rerun()

        # ── Main tabbed interface ─────────────
        tabs = st.tabs([
            "📋 Spreadsheet",
            "ƒ Formulas",
            "🧹 Data Cleaning",
            "📊 Analytics",
            "🎨 Formatting",
            "🔍 Sort & Filter",
            "📈 Charts",
            "🤖 AI Assistant",
            "💾 Export",
        ])

        with tabs[0]:
            self._render_toolbar(df)
            df = self._active_df()
            self._render_data_grid(df)

        with tabs[1]:
            self._render_formula_bar(self._active_df())

        with tabs[2]:
            self._render_data_cleaning(self._active_df())

        with tabs[3]:
            self._render_analytics(self._active_df())

        with tabs[4]:
            self._render_conditional_formatting(self._active_df())

        with tabs[5]:
            self._render_sorting_filtering(self._active_df())

        with tabs[6]:
            self._render_charts(self._active_df())

        with tabs[7]:
            self._render_ai_assistant(self._active_df())

        with tabs[8]:
            self._render_export(self._active_df())
