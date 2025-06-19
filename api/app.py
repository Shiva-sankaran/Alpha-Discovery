# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sdk.backtester import run_backtest
from register_signal import register_signal, get_signal
import numpy as np

app = FastAPI()


def to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class BacktestRequest(BaseModel):
    symbol: str
    signal_name: str

class RegisterSignalRequest(BaseModel):
    signal_name: str
    function_code: str

@app.post("/register_signal/")
def register_signal_handler(request: RegisterSignalRequest):
    success = register_signal(request.signal_name, request.function_code)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to register signal.")
    return {"message": f"Signal '{request.signal_name}' registered successfully."}

@app.post("/backtest/")
def backtest_handler(request: BacktestRequest):
    try:
        df = pd.read_feather(f"features/{request.symbol}.feather").set_index("date")
        signal_func = get_signal(request.signal_name)
        if signal_func is None:
            raise HTTPException(status_code=404, detail="Signal not found.")
        result = run_backtest(request.symbol, df, request.signal_name)
        result_serializable = {k: to_serializable(v) for k, v in result.items()}
        return result_serializable
    except Exception as e:
        import traceback
        traceback.print_exc()  # 👈 This prints full stack trace to console
        raise HTTPException(status_code=500, detail=str(e))

