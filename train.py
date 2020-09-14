#!/usr/bin/python

import time
import torch
from options.train_options import TrainOptions
from data import CreateDataLoader
from util.visualizer import Visualizer
from models import create_model



def train(train_options):
    opt = train_options
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size, flush=True)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    # print(torch.cuda.memory_allocated())

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            if i % 100 == 0:
                print(str(i) + ' trained in this epoch so far', flush=True)
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps), flush=True)
                model.save_networks('latest')

            iter_data_time = time.time()
            torch.cuda.empty_cache()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps), flush=True)
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time), flush=True)
        model.update_learning_rate()
#
# if __name__ == '__main__':
#     # opt = TrainOptions().parse()
#     data_loader = CreateDataLoader(opt)
#     dataset = data_loader.load_data()
#     dataset_size = len(data_loader)
#     print('#training images = %d' % dataset_size, flush=True)
#
#     model = create_model(opt)
#     model.setup(opt)
#     visualizer = Visualizer(opt)
#     total_steps = 0
#     #print(torch.cuda.memory_allocated())
#
#     for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
#         epoch_start_time = time.time()
#         iter_data_time = time.time()
#         epoch_iter = 0
#
#         for i, data in enumerate(dataset):
#             if i % 100 == 0:
#                 print(str(i) + ' trained in this epoch so far', flush=True)
#             iter_start_time = time.time()
#             if total_steps % opt.print_freq == 0:
#                 t_data = iter_start_time - iter_data_time
#             visualizer.reset()
#             total_steps += opt.batch_size
#             epoch_iter += opt.batch_size
#             model.set_input(data)
#             model.optimize_parameters()
#
#             if total_steps % opt.display_freq == 0:
#                 save_result = total_steps % opt.update_html_freq == 0
#                 visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
#
#             if total_steps % opt.print_freq == 0:
#                 losses = model.get_current_losses()
#                 t = (time.time() - iter_start_time) / opt.batch_size
#                 visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
#                 if opt.display_id > 0:
#                     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
#
#             if total_steps % opt.save_latest_freq == 0:
#                 print('saving the latest model (epoch %d, total_steps %d)' %
#                       (epoch, total_steps), flush=True)
#                 model.save_networks('latest')
#
#             iter_data_time = time.time()
#             torch.cuda.empty_cache()
#         if epoch % opt.save_epoch_freq == 0:
#             print('saving the model at the end of epoch %d, iters %d' %
#                   (epoch, total_steps), flush=True)
#             model.save_networks('latest')
#             model.save_networks(epoch)
#
#         print('End of epoch %d / %d \t Time Taken: %d sec' %
#               (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time), flush=True)
#         model.update_learning_rate()
