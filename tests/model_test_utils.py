from collections import Counter

# https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
def get_layers(model):
    children = list(model.children())
    flatt_children = []
    if children == []:
        return model
    else:
       for child in children:
            try:
                flatt_children.extend(get_layers(child))
            except TypeError:
                flatt_children.append(get_layers(child))
    return flatt_children

def get_model_layer_counts(layers):
    layers_type = [x.__class__.__name__ for x in layers]
    layers_count = Counter(layers_type)

    return layers_count

def count_params(model):
    return sum(p.numel() for p in model.parameters())