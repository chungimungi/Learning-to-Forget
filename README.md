# Learning-to-Forget

This repository attempts to implement in code a model that learns to forget. By implementing a simple text classifier model on a **wine-review** dataset, randomizing the weights of a trained model can be used as a method to make a trained model forget what it has learnt. 


### Model Parameters


| Layer (type:depth-idx)              | Param   |
|-------------------------------------|---------|
| TextClassifier                      | --      |
| └─Linear: 1-1                       | 356,160 |
| └─Linear: 1-2                       | 19,110  |
| └─ReLU: 1-3                         | --      |
| └─Dropout: 1-4                      | --      |

Total params: 375,270  
Trainable params: 375,270  
Non-trainable params: 0  

### Results

**Original Model**

|                      |                      |
|----------------------|----------------------|
| Loss                 | 0.6329621076583862   |
| Train Accuracy       | 0.8930568099021912   |
| Validation Loss      | 0.6247082948684692   |
| Validation Accuracy  | 0.9080535769462585   |




 ![image](https://github.com/chungimungi/Learning-to-Forget/assets/90822297/d1a0edd5-031c-40ea-b8b3-ba3bad7a411a)

 ![image](https://github.com/chungimungi/Learning-to-Forget/assets/90822297/b4c1a0b2-c44c-45c1-9517-f2c20dc7993a)


**Randomized Weights Model i.e the model that forgot**

|                      |                      |
|----------------------|----------------------|
| Loss                 | 3.1483829021453857   |
| Train Accuracy       | 0.507146954536438    |
| Validation Loss      | 3.06455397605896     |
| Validation Accuracy  | 0.508832573890686    |



 ![image](https://github.com/chungimungi/Learning-to-Forget/assets/90822297/a578d38f-bd75-4abb-987f-78ba6a252c95)

 ![image](https://github.com/chungimungi/Learning-to-Forget/assets/90822297/2ddb92e4-ec72-4de4-86a6-6bc43f81b47f)


**Comparison Graph**

![image](https://github.com/chungimungi/Learning-to-Forget/assets/90822297/dfe2a9ce-eae5-4cb5-b258-e89084bd75f4)


