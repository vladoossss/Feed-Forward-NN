

1) data
    * data generator
    * preprocessing
    * train_test
    * data loader
    * feature generation
    
2) nn
    * count of layers
    * count of neurons
    * activation functions
    
3) trainer
    * optimizer
    * parameters(learning_rate,...)
    * count of epochs
    * batch size
    * loss function
    * metrics
    * save/load
    
* В папке loss_img находятся последовательные изменения функции потерь
  в ходе обучения на тренировочной выборке на каждой из эпох.
  
* В папке cm_img находятся последовательные изменения 
  матриц ошибок для каждого батча из тестового множества.

* В папке class_img находятся последовательные изменения визуализации
  результатов классификации для каждого батча из тестового множества.
  
В каждой из 3 папок также присутствует итоговое изображение в формате .gif
    
