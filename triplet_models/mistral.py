import fireworks.client
models = fireworks.client.Model.list()

model_names =models.data
for model in model_names:
    print(model.id)
