from w_net_v11 import get_unet
from data_loader import get_data_generators
from keras.callbacks import TensorBoard, ModelCheckpoint, ProgbarLogger
import os


def main(args):
    img_rows = 192
    img_cols = 336
    batch_size = 6
    n_epochs = 100
    models_folder = 'models'
    model_name = 'w_net_V12'
    model_path = os.path.join(models_folder, model_name)

    train_generator, val_generator, training_samples, val_samples = get_data_generators(train_folder='train',
                                                                                        val_folder='validation',
                                                                                        img_rows=img_rows,
                                                                                        img_cols=img_cols,
                                                                                        batch_size=batch_size)

    print('found {} training samples and {} validation samples'.format(training_samples, val_samples))
    print('...')
    print('building model...')

    w_net, disp_map_model = get_unet(img_rows=img_rows, img_cols=img_cols, lr=1e-4)

    print('saving model to {}...'.format(model_path))
    model_yaml = w_net.to_yaml()
    with open(model_path + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    print('begin training model, {} epochs...'.format(n_epochs))
    for epoch in range(n_epochs):

        print('epoch {} \n'.format(epoch))

        model_path = os.path.join(models_folder, model_name + '_epoch_{}'.format(epoch))
        w_net.fit_generator(train_generator,
                            steps_per_epoch=training_samples // batch_size,
                            epochs=1,
                            validation_data=val_generator,
                            validation_steps=val_samples // batch_size,
                            verbose=1,
                            callbacks=[TensorBoard(log_dir='/tmp/deepdepth'),
                                       ModelCheckpoint(model_path + '.h5', monitor='loss',
                                                       verbose=0,
                                                       save_best_only=False,
                                                       save_weights_only=True,
                                                       mode='auto', period=1)])

if __name__ == '__main__':
    main(None)
