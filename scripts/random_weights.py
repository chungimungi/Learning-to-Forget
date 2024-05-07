def randomize_weights(model):
    for param in model.parameters():
        if param.requires_grad:
            param.data = torch.randn_like(param)

randomize_weights(initial_model)