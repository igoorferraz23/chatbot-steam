import pandas as pd


class Tools:
    @staticmethod
    def safe_int(valor) -> int:
        """Garante a conversão dos valores para int"""
        try:
            v = float(valor)
            if pd.isna(v) or v == float("inf") or v == float("-inf"):
                return 0
            return int(v)
        except (ValueError, TypeError):
            return 0
