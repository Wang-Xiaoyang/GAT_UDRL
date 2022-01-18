```
def pre_training (R, s, a, t, masked_a): 
    # R: reward-to-go, not masked
    # s: states, not masked
    # a: actions, 15% masked in each sequence
    # t: timestep; may not be used
    # compute embeddings
    s_embedding = embed_s(s)
    R_embedding = embed_R(R)
    a_embedding = embed_a(a)
    attention_mask = att_mask() # to figure out    
    input_embeds = stack(R_embedding , s_embedding , a_embedding)
    logits_lm = Encoder(input_embeds)
    return logits_lm
    
for (R, s, a, t, masked_a) in dataloader:
    logits_lm = pre_train(R, s, a, t, masked_a)
    loss = CrossEntropy(logis_lm, masked_a)
    optimizer.zero_grad ()
    loss.backward()
    optimizer.step()
```
