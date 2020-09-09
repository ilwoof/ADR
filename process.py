#%%
from pathlib import Path

from ADR.preprocess import *
from ADR.ADR import *


log_path = Path('data/Drain_result/bgl_385events/ECM_sessions.npz')
X_Y = np.load(log_path)

#%%
x, y = X_Y['df_X'][:10000, :], X_Y['df_Y'][:10000]

#%%
np.savez("data/Drain_result/bgl_385events/ECM_sessions_10k.npz", df_X=x, df_Y=y)