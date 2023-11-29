import json


from json import JSONEncoder
from typing import List


class dto:
    __attribute__: List[str]
    __attribute_type__: List[object]

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                attr_index = self.__attribute__.index(key)
            except ValueError:
                continue
            attr_type = self.__attribute_type__[attr_index]
            if not isinstance(value, list):
                if dto in attr_type.__bases__:
                    setattr(self, key, attr_type(**value))
                else:
                    setattr(self, key, attr_type(value))
                continue
            if dto in attr_type.__bases__:
                setattr(self, key, [attr_type(**elem) for elem in value])
            else:
                setattr(self, key, [attr_type(elem) for elem in value])

    def to_json(self):
        result = dict()
        for elem in self.__attribute__:
            if hasattr(self, elem):
                result[elem] = self.__getattribute__(elem)
        return result

    def __str__(self) -> str:
        return json.dumps(self.to_json(), indent=4, cls=DTOEncoder)


class DTOEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, dto) or hasattr(o, "to_json"):
            return o.to_json()
        return o.__dict__


class label(dto):
    id: str
    output: str
    __attribute__ = ["id", "output"]
    __attribute_type__ = [str, str]


class labels(dto):
    task: str
    golds: List[label]
    __attribute__ = ["task", "golds"]
    __attribute_type__ = [str, label]
