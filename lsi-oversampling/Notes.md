Given a batch `b_orig` of `n` images, we want a new batch `b_new` of `n` images, each of which is obtained by a weighted sum of all the images in `b_orig`, with a different set of weights applied for each resultant image in  `b_new`.

The set of weights `wts` will be of the shape `[n n]` since each new image is a weighted sum of all the original images. If we generate a random set of weights (however, this is not the case for us), we will need to normalize them so that their row-wise sum for all rows equals `1`. This can be done using PyTorch's inbuilt `normalize()` function.

```
import torch
import torch.nn.functional as F

wts = torch.rand(128, 128)
wts_norm = F.normalize(wts, p=1, dim=1)

wts_norm[0].sum()
# Should return 1.
```

The choice of `p` above is for normalizing according to the L_p norm. Since we want the sum of these weights, we use the L_1 norm. `dim` is set to `1` since we want row-wise sum to be `1`.

Now, we generate the weighted sum of the images. In order to multiply along the axis of images, we need to permute the axes.

```
# b_orig -> [n, c, h, w]
b_orig_perm = b_orig.permute(1,2,3,0)
# b_orig_perm -> [c, h, w, n]

b_new = b_orig_perm @ wts_norm.t()
# For our purpose, this can just be wts instead of wts_norm.
```

[PyTorch] If I have to compute a loss between pairs of ground truth and predictions, and there is just one prediction and multiple ground truths, what is the best way to compute them? This is for a classification task, and the loss is cross-entropy. 

Imagine the prediction is a c-element vector, and there are k ground truth labels, so I would need to calculate loss(pred, gt_i) for all i from 1 to k. In the end, I want a k-element loss vector.

The brute force way of doing this is to calculate the loss above k times. I was wondering if there is a more efficient way, maybe using broadcasting? I can broadcast the predictions to be of the shape [k, c], but how can I calculate the same function for each row pair between this broadcasted prediction and the ground truths?