https://www.runoob.com/w3cnote/yaml-intro.html

- 一种标记语言
- 大小写敏感
- 使用缩进表示层级关系
- 缩进不允许使用tab，只允许空格
- 缩进的空格数不重要，只要相同层级的元素左对齐即可
- '#'表示注释

## 数据类型:
- 对象
- 数组
- 纯量

### 对象
对象键值对使用冒号结构表示 key: value，冒号后面要加一个空格。

也可以使用 `key:{key1: value1, key2: value2, ...}`。

还可以使用缩进表示层级关系；
```yaml
key: 
    child-key: value
    child-key2: value2
较为复杂的对象格式，可以使用问号加一个空格代表一个复杂的 key，配合一个冒号加一个空格代表一个 value：

?  
    - complexkey1
    - complexkey2
:
    - complexvalue1
    - complexvalue2
意思即对象的属性是一个数组 [complexkey1,complexkey2]，对应的值也是一个数组 [complexvalue1,complexvalue2]
```

### 数组
以 - 开头的行表示构成一个数组：
```yaml
- A
- B
- C
```
```yaml
key: [value1, value2, ...]
```
多维数组
```yaml
-
 - A
 - B
 - C
```
一个相对复杂的例子：
```yaml
companies:
    -
        id: 1
        name: company1
        price: 200W
    -
        id: 2
        name: company2
        price: 500W

```
意思是 companies 属性是一个数组，每一个数组元素又是由 id、name、price 三个属性构成。

数组也可以使用流式(flow)的方式表示：

companies: [{id: 1,name: company1,price: 200W},{id: 2,name: company2,price: 500W}]
复合结构
数组和对象可以构成复合结构，例：
```yaml
languages:
  - Ruby
  - Perl
  - Python 
websites:
  YAML: yaml.org 
  Ruby: ruby-lang.org 
  Python: python.org 
  Perl: use.perl.org
转换为 json 为：

{ 
  languages: [ 'Ruby', 'Perl', 'Python'],
  websites: {
    YAML: 'yaml.org',
    Ruby: 'ruby-lang.org',
    Python: 'python.org',
    Perl: 'use.perl.org' 
  } 
}
```
### 引用
`& 用来建立锚点（defaults），<< 表示合并到当前数据，* 用来引用锚点。`
```yaml
defaults: &defaults
  adapter:  postgres
  host:     localhost

development:
  database: myapp_development
  <<: *defaults

test:
  database: myapp_test
  defaults: *defaults
```
等同于
```python
{'defaults': {'adapter': 'postgres', 'host': 'localhost'},
 'development': {'adapter': 'postgres',
                 'database': 'myapp_development',
                 'host': 'localhost'},
 'test': {'database': 'myapp_test',
          'defaults': {'adapter': 'postgres', 'host': 'localhost'}}}
```

```yaml
- &showell Steve 
- Clark 
- Brian 
- Oren 
- *showell 
```
等同于 `[ 'Steve', 'Clark', 'Brian', 'Oren', 'Steve' ]`