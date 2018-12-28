# Labelme

## 2018.12.11

### canvas.py
#### Line88 createMode
在createMode中加入auto
```python
    def createMode(self, value):
        if value not in ['polygon', 'rectangle', 'line', 'point', 'auto']:
            raise ValueError('Unsupported createMode: %s' % value)
        self._createMode = value
```
#### Line 190 mouseMoveEvent
autoMode移动鼠标时产生矩形框
```python
    elif self.createMode == 'auto':
        self.line.points = list(self.getRectangleFromLine(
            (self.current[0], pos)
        ))
        self.line.close()
```

#### Line 231 _init_
加入action autoMode

#### Line 287 mousePressEvent
加入automode

```python
    elif self.createMode == 'auto':
        assert len(self.current.points) == 1
        self.current.points = self.line.points
        self.line.auto_segment()
```

### app.py
#### Line 231  init  ！！！
在menu和actions中加入automode
现在的问题是，shortcuts快捷键找不到位置，对应功能位置没有找到
```python
autoMode = action(
    'Auto Select',
    lambda: self.toggleDrawMode(False, createMode='auto'),
    # shortcuts['auto_select'],
    'objects',
    'Start drawing rectangle',
    enabled=True
        )
```
#### Line 537 populateModeActions
加入 automode
```python
actions = (
    self.actions.createMode,
    self.actions.createRectangleMode,
    self.actions.createLineMode,
    self.actions.createPointMode,
    self.actions.editMode,
    self.actions.autoMode
        )
```

#### Line 561 setClean
加入 autoMode
```python
self.actions.autoMode.setEnabled(True)
```

#### Line 626 toggleDrawMode
加入autoMode使能状态
```
if edit:
    self.actions.createMode.setEnabled(True)
    self.actions.createRectangleMode.setEnabled(True)
    self.actions.createLineMode.setEnabled(True)
    self.actions.createPointMode.setEnabled(True)
    self.actions.autoMode.setEnabled(True)
else:
    if createMode == 'polygon':
        self.actions.createMode.setEnabled(False)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.autoMode.setEnabled(True)
    elif createMode == 'rectangle':
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(False)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.autoMode.setEnabled(True)
    elif createMode == 'line':
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(False)
        self.actions.createPointMode.setEnabled(True)
        self.actions.autoMode.setEnabled(True)
    elif createMode == 'point':
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(False)
        self.actions.autoMode.setEnabled(True)
    elif createMode == 'auto':
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.autoMode.setEnabled(False)
```
### shape.py
#### Line 188 auto_segment
通过矩形框选取点,并将定点坐标保存到json文件中
```python
    def auto_segment(self):
        point0 = self.points[0]
        point1 = self.points[1]
        roi = [point0.x(), point0.y(), point1.x(), point1.y()]
        json.dump(roi, open('../output.json', 'w'), indent=2)
```

Image位置: 
app.py  1019
self.image=image

## 2018.12.12
###canvas.py
#### Line309 mousePressEvent
将选取的点作为roi读取进来

### app.py
#### Line1131 openFile
在openfile时把地址传入self.imagePsath

## 2018.12.13
### canvas.py
#### Line 315 mousePressEven
从 app.py 传入 filename,按选取的roi送入deeplab中进行检测