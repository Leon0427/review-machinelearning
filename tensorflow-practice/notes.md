##### session
```python
sess.run(tf.global_variables_initializer()) # 初始化
batch_loss, _, training_step = sess.run(
                    [loss_tensor, train_op, global_step],
                    feed_dict={image_place:train_batch_data,
                                label_place:train_batch_label,
                                dropout_param:0.5})
# 计算loss_tensor, train_op, global_step, 为variable输送数据,返回值分别是计算对象的计算结果
```
##### saver
```python
saver = tf.train.Saver()
with sess.as_default():
    saver.restore(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix)) # 读取模型
    save_path = saver.save(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix)) # 存储模型
```
