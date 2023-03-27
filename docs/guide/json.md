# JSON Interoperability

Many of the classes and objects in Brush can be serialized to and from json. 
This is achieved using the [cpp json](https://github.com/nlohmann/json) package. 
It means you can, for example, write programs as Python dictionaries like so:


```python

json_program = {
    "Tree": [
        { "node_type":"Add", "is_weighted": True },
        { "node_type":"Terminal", "feature":"x1"},
        { "node_type":"Terminal", "feature":"x2"}
    ],
    "is_fitted_":False
}

```

This program is a weighted addition of `x1` and `x2`, i.e., $$c_1 x_1 + c_2 x_2$$. 
Note the dictionary must contain keys for all members of the Program class, `Tree` and `is_fitted_`. 
We can turn this directly into a regressor: 

```python

PRG = Regressor(json_program)
print( "program:", PRG.get_model())

```