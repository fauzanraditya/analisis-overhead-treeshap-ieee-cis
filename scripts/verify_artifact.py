import json
import numpy as np
import pandas as pd
from model_registry import load_model,validate

result={'purpose':'synthetic artifact smoke test, not predictive evaluation','models':{}}
for mid in ['M1','M2']:
    pre,clf,exp,features,actual=load_model(mid)
    result['models'][mid]={'structure':actual,**validate(pre,clf,exp,pd.DataFrame(np.zeros((2,len(features)),dtype=np.float32),columns=features))}
print(json.dumps(result,indent=2))
