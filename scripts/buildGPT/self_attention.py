import tensorflow as tf

# tril in pytorch means lower triangular matrix
# to do the same in tensorflow, we can use tf.linalg.band_part
a = tf.linalg.band_part(tf.ones([3, 3]), -1, 0)
a = a / tf.reduce_sum(a, axis=-1, keepdims=True)
print("a =")
print(a)

b = tf.random.uniform(shape=(3, 2),
                      minval=0,
                      maxval=10,
                      dtype=tf.float32)
print("b =")
print(b)

c = a @ b
print("c =")
print(c)

# set my seed
tf.random.set_seed(69)
B, T, C = 4, 8, 2
x = tf.random.normal([B, T, C])
print("x =")
print(x)

# We want x[b, t] = mean_{i<=t} x[b, i]
xbow = tf.zeros([B, T, C])
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = tf.reduce_mean(xprev, axis=0)
