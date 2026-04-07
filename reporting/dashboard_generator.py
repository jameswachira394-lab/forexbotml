"""
reporting/dashboard_generator.py
--------------------------------
Generates an interactive HTML dashboard using actual project data.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DashboardGenerator:
    def __init__(self, symbol: str, output_dir: str = "reports"):
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.model_meta_path = Path("saved_models") / f"{symbol}_forex_xgb_meta.json"
        self.trades_path     = Path("logs") / f"{symbol}_backtest_trades.csv"
        self.equity_path     = Path("logs") / f"{symbol}_equity_curve.csv"

    def generate(self):
        """Main entry point to generate the HTML file."""
        logger.info(f"[{self.symbol}] Generating dashboard...")
        
        # 1. Load Data
        meta   = self._load_meta()
        trades = self._load_trades()
        equity = self._load_equity()
        
        # 2. Process Data for Template
        data_map = self._process_data(meta, trades, equity)
        
        # 3. Read Template & Inject
        template = self._get_template()
        html = self._inject_data(template, data_map)
        
        # 4. Save
        output_path = self.output_dir / f"{self.symbol}_dashboard.html"
        output_path.write_text(html, encoding="utf-8")
        logger.info(f"[{self.symbol}] Dashboard saved -> {output_path}")
        return output_path

    def _load_meta(self) -> dict:
        if not self.model_meta_path.exists():
            logger.warning(f"Metadata not found: {self.model_meta_path}")
            return {}
        with open(self.model_meta_path) as f:
            return json.load(f)

    def _load_trades(self) -> pd.DataFrame:
        if not self.trades_path.exists():
            logger.warning(f"Trade log not found: {self.trades_path}")
            return pd.DataFrame()
        return pd.read_csv(self.trades_path)

    def _load_equity(self) -> pd.DataFrame:
        if not self.equity_path.exists():
            logger.warning(f"Equity curve not found: {self.equity_path}")
            return pd.DataFrame()
        return pd.read_csv(self.equity_path)

    def _process_data(self, meta, trades, equity) -> dict:
        """Map raw data to the dashboard's JS structures."""
        
        # Default KPIs
        stats = {
            "symbol": self.symbol,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "total_trades": 0,
            "profit_factor": 0.0,
            "max_dd": 0.0,
            "auc": meta.get("cv_auc_mean", 0.0),
            "test_auc": meta.get("test_auc", 0.0),
            "is_synthetic": "Synthetic Data" if "synthetic" in str(self.symbol).lower() else "Real Data"
        }

        # 1. Trade Analytics
        trade_list = []
        pnl_data = [0]*13 # bins for -600 to 600+
        if not trades.empty:
            stats["total_trades"] = len(trades)
            stats["win_rate"] = (trades["pnl_usd"] > 0).mean()
            stats["net_pnl"] = trades["pnl_usd"].sum()
            
            gross_p = trades[trades["pnl_usd"] > 0]["pnl_usd"].sum()
            gross_l = abs(trades[trades["pnl_usd"] < 0]["pnl_usd"].sum())
            stats["profit_factor"] = gross_p / max(gross_l, 1e-6)
            
            # First 20 trades for the table
            for _, r in trades.head(20).iterrows():
                trade_list.append([
                    str(r["entry_ts"])[:16],
                    "LONG" if r["direction"] > 0 else "SHORT",
                    float(r["pnl_usd"]),
                    str(r["exit_reason"]),
                    float(r["ml_prob"])
                ])
            
            # PnL distribution
            bins = [-600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600]
            counts, _ = np.histogram(trades["pnl_usd"].fillna(0), bins=bins + [999999])
            pnl_data = counts.tolist()

        # 2. Equity Curve
        equity_data = []
        if not equity.empty:
            # Downsample if too many points (keep ~500 max)
            if len(equity) > 500:
                indices = np.linspace(0, len(equity) - 1, 500).astype(int)
                equity = equity.iloc[indices]
            
            for _, r in equity.iterrows():
                equity_data.append([str(r["timestamp"])[:10], float(r["equity"])])
            
            # Max DD
            peak = 0
            max_dd = 0
            for val in equity["equity"]:
                if val > peak: peak = val
                dd = (peak - val) / peak if peak > 0 else 0
                if dd > max_dd: max_dd = dd
            stats["max_dd"] = max_dd

        # 3. Model Metrics
        fi_list = []
        fi_dict = meta.get("feature_importance", {})
        if fi_dict:
            # Sort and take top 20
            sorted_fi = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)[:20]
            fi_list = [[k, float(v)] for k, v in sorted_fi]
            
        prob_buckets = meta.get("prob_buckets", [])
        
        return {
            "stats": stats,
            "equityData": equity_data,
            "features": fi_list,
            "probBuckets": prob_buckets,
            "pnlBuckets": {"labels": ['-600','-500','-400','-300','-200','-100','0','100','200','300','400','500','600+'], "data": pnl_data},
            "trades": trade_list,
            "threshold": meta.get("threshold", 0.5)
        }

    def _inject_data(self, template: str, data: dict) -> str:
        s = data["stats"]
        
        # Replace Header/KPI placeholders
        html = template
        html = html.replace("XGBoost · EURUSD M5 · Jan 2022 – Jul 2023", f"XGBoost · {s['symbol']} · Result Dashboard")
        html = html.replace("AUC 0.769", f"AUC {s['auc']:.3f}")
        html = html.replace("57% Win Rate", f"{s['win_rate']:.1%} Win Rate")
        html = html.replace("Synthetic Data", s["is_synthetic"])
        
        # Page 1 KPIs
        html = html.replace("+$21,786", f"${s['net_pnl']:+,.0f}")
        html = html.replace("57.0%", f"{s['win_rate']:.1%}")
        html = html.replace("1.86", f"{s['profit_factor']:.2f}")
        html = html.replace("-5.65%", f"-{s['max_dd']:.1%}")
        html = html.replace("270", str(s["total_trades"]))
        
        # Inject JS Data
        # We look for the 'const equityData = [...];' and similar blocks
        html = self._replace_js_var(html, "equityData", data["equityData"])
        html = self._replace_js_var(html, "features", data["features"])
        html = self._replace_js_var(html, "probBuckets", data["probBuckets"])
        html = self._replace_js_var(html, "pnlBuckets", data["pnlBuckets"])
        html = self._replace_js_var(html, "trades", data["trades"])
        
        return html

    def _replace_js_var(self, html: str, var_name: str, data: any) -> str:
        import re
        pattern = rf"const {var_name} = \[.*?\n\];"
        if var_name == "pnlBuckets":
            pattern = rf"const {var_name} = \{{.*?\n\}};"
            
        replacement = f"const {var_name} = {json.dumps(data, indent=2)};"
        
        # If the pattern is not found with multiline search, we might need a more flexible one
        # The user's template has some complex line breaks.
        # Let's try a simpler replace for the specific block.
        if f"const {var_name} = [" in html or f"const {var_name} = {{" in html:
            # We'll use a more robust regex that finds the variable assignment
            search_pattern = rf"const {var_name}\s*=\s*(?:\[|\{{)[\s\S]*?(?:\]|\}})\s*(?=;)"
            html = re.sub(search_pattern, f"const {var_name} = {json.dumps(data, indent=2)}", html)
            
        return html

    def _get_template(self) -> str:
        # Use the provided HTML as the base
        # I'll save a copy of the user's HTML to a template file first if I had it.
        # Since I have the content in the prompt, I'll hardcode it or read from a local file if we saved it.
        # For this execution, I'll expect a file named 'reporting/template.html' to exist.
        template_path = Path("reporting/template.html")
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")
        else:
            # Fallback to a very minimal version or raise error
            raise FileNotFoundError("Dashboard template not found at reporting/template.html")

