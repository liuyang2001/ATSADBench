DATA_PATH = "dataset/"
NORMAL_DATA_FIRST_SIX = "M-train_data.xlsx"
NORMAL_DATA_SEVENTH_NINE = "U-train_data.xlsx"
MULTI_VAR_COLUMNS = ['Variable 1','Variable 2','Variable 3','Variable 4','Variable 5', 'Variable 6', 'Variable 7', 'Variable 8','Variable 9','Variable 10','Variable 11','Variable 12','Variable 13','Variable 14','Variable 15','Variable 16','Variable 17','Variable 18','Variable 19','Variable 20','Variable 21','Variable 22','Variable 23','Variable 24','Variable 25','Variable 26','Variable 27']
WINDOW_SIZE = 20  
STEP_SIZE = 5     
HORIZON = STEP_SIZE
ALPHA = 0.2        
BETA = 0.1         
num_samples = 1
POSITIVE_SAMPLE_NUMBER = 0
RAG_NUMBER= 0
stage = "process"
TEST_DATA_PATTERN = f"_{WINDOW_SIZE}_test_data.xlsx"
MODEL_CONFIG = {
    "deepseek-chat": {"api_key": "", "endpoint": "https://api.deepseek.com/v1"}, 
    "qwen3-235b-a22b":{"api_key": "", "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1"}
}
COLUMNS_MIN_MAX_VALUES={
    "Variable 1": {
        "min": 0,
        "max": 0
    },
    "Variable 2": {
        "min": 0,
        "max": 0
    },
    "Variable 3": {
        "min": 0,
        "max": 0
    },
    "Variable 4": {
        "min": 0,
        "max": 0
    },
    "Variable 5": {
        "min": 0,
        "max": 0
    },
    "Variable 6": {
        "min": 0,
        "max": 0
    },
    "Variable 7": {
        "min": 0,
        "max": 0
    },
    "Variable 8": {
        "min": 0,
        "max": 0
    },
    "Variable 9": {
        "min": 0,
        "max": 0
    },
    "Variable 10": {
        "min": 0,
        "max": 0
    },
    "Variable 11": {
        "min": 0,
        "max": 0
    },
    "Variable 12": {
        "min": 0,
        "max": 0
    },
    "Variable 13": {
        "min": 0,
        "max": 0
    },
    "Variable 14": {
        "min": 0,
        "max": 0
    },
    "Variable 15": {
        "min": 0,
        "max": 0
    },
    "Variable 16": {
        "min": 0,
        "max": 0
    },
    "Variable 17": {
        "min": 0,
        "max": 0
    },
    "Variable 18": {
        "min": 0,
        "max": 0
    },
    "Variable 19": {
        "min": 0,
        "max": 0
    },
    "Variable 20": {
        "min": 0,
        "max": 0
    },
    "Variable 21": {
        "min": 0,
        "max": 0
    },
    "Variable 22": {
        "min": 0,
        "max": 0
    },
    "Variable 23": {
        "min": 0,
        "max": 0
    },
    "Variable 24": {
        "min": 0,
        "max": 0
    },
    "Variable 25": {
        "min": 0,
        "max": 0
    },
    "Variable 26": {
        "min": 0,
        "max": 0
    },
    "Variable 27": {
        "min": 0,
        "max": 0
    }
}