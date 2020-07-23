### Deep Matrix Factorization Models for Recommender Systems

A Non-official Implementation of "Deep Matrix Factorization Models for Recommender Systems"

See paper: http://www.ijcai.org/proceedings/2017/0447.pdf

Based on implementation: https://github.com/hegongshan/deep_matrix_factorization 

### Environment Settings

We use Keras with Tensorflow as the backend.

- Keras version: 2.3.0
- TensorFlow: 2.0.0 

### Example to run the codes.

```
python dmf.py --dataset retail-rocket --user_layers [512,64] --item_layers [1024,64] --epochs 100 --lr 0.0001
```

