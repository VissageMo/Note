# Python 语法笔记

## 类初始化

```python
class Base():
    def __init__(self, A, B):
        self.A = A
        self.B = B
        c = self.base(A, B)

    def base(self, a, b):
        return a * b

Base(1, 2)
```

使用class定义类时，需通过