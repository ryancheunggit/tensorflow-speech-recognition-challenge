import pandas as pd
import os

def parse_log(filepath):
    # print(filepath)
    model_name = '_'.join(filepath.split('/')[2].split('.md')[0].split())
    print(" ---- start process log for model {}".format(model_name))
    lines = open(filepath).readlines()
    fold_num = 0
    fitting_times = []
    loss_progress = []
    learning_rate_decay_rounds = []
    total_params = 0
    for line_num, line_content in enumerate(lines):
        if line_content.startswith('Total params:'):
            total_params = line_content.split(': ')[1]
            # print(" --- total_params {}".format(total_params))
        if line_content.startswith('========= fitting'):
            _, _, fold_num, _, _, start_time, _ = line_content.split()
            start_time = pd.to_datetime(start_time)
            # print(" --- start_time {}".format(start_time))
            epoch = 0
        if line_content.startswith('========= generating oof'):
            end_time = pd.to_datetime(line_content.split()[-2])
            if end_time < start_time:
                end_time += pd.Timedelta(1, unit='D')
            # print(" --- end_time {}".format(end_time))
            fitting_times.append(pd.Series({
                'fitting_time':end_time - start_time
            }))
        if '[==============================]' in line_content:
            epoch += 1
            train_loss = line_content.split(' loss: ')[1].split()[0]
            train_acc = line_content.split(' categorical_accuracy: ')[1].split()[0]
            valid_loss = line_content.split('val_loss: ')[1].split()[0]
            valid_acc = line_content.split(' val_categorical_accuracy: ')[1].split()[0]
            loss_progress.append(pd.Series({
                'fold': fold_num,
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_acc': train_acc,
                'valid_acc': valid_acc
            }))
        if 'ReduceLROnPlateau' in line_content:
            learning_rate_decay_rounds.append(pd.Series({
                'fold': fold_num,
                'epoch': epoch,
                'lr': line_content.replace('.', '').split()[-1]
            }))
    pd.DataFrame(loss_progress).to_csv('../Presentation/loss_progress_{}.csv'.format(model_name), index=False)
    pd.DataFrame(learning_rate_decay_rounds).to_csv('../Presentation/lr_decay_{}.csv'.format(model_name), index=False)
    pd.DataFrame(fitting_times).to_csv('../Presentation/fitting_times_{}.csv'.format(model_name), index=False)

files = os.listdir('../Presentation/')
filepaths = ['../Presentation/{}'.format(filename) for filename in files if filename.endswith('.md')]

for filepath in filepaths:
    parse_log(filepath)
