
import json

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
import scipy

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from skimage.transform import resize

def plot_various_models_RA_rot(nn_scs_RA, nn_scs_m_RA, nns_RA, nns_m_RA, bis_RA, bis_m_RA, gts_RA, right_angles, keep_arr = [16,17,18,19]):
    
    rot_arr_RA_sc = []
    rot_arr_RA_sc_m = []

    rot_arr_RA_RA = []
    rot_arr_RA_RA_m = []


    rot_arr_RA_bi = []
    rot_arr_RA_bi_m = []

    rot_gt = []

    angle_title = []
    for i in range(24):
        if i in keep_arr:
            rot_arr_RA_sc.append(nn_scs_RA[i])
            rot_arr_RA_sc_m.append(nn_scs_m_RA[i])
            
            rot_arr_RA_RA.append(nns_RA[i])
            rot_arr_RA_RA_m.append(nns_m_RA[i])
            
            rot_arr_RA_bi.append(bis_RA[i])
            rot_arr_RA_bi_m.append(bis_m_RA[i])
            
            rot_gt.append(gts_RA[i])
            
            angle_title.append(right_angles[i])
            


    start_id = 0
    model_used = ['nnUNet_SC', 'nnUNet', 'Bispectral']

    nbr_rot = len(model_used)
    nbr_model = len(rot_gt)
    fig = plt.figure(figsize=(6*nbr_model,5*nbr_rot))
    show_diff = False
    show_softmax = True
    cntr = 0
    for current_rot in range(len(rot_gt)):
        if 'nnUNet_SC' in model_used:
            cntr += 1
            ax = fig.add_subplot(nbr_rot, nbr_model,cntr, projection='3d')
            angle_ = f'nnUNet_SC {angle_title[current_rot][0]}z {angle_title[current_rot][1]}y {angle_title[current_rot][2]}z'
            ax.set_title(angle_)
            
            if show_softmax:
                nn_scs_val = rot_arr_RA_sc_m[current_rot] * (rot_arr_RA_sc[current_rot]>0).astype(int)
                gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z = np.nonzero(nn_scs_val)
                cs = nn_scs_val[gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z]
                c=scalarMap.to_rgba(cs)
                ax.scatter(gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z, c=c)
            else:
                gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z = np.nonzero(rot_arr_RA_sc[current_rot])
                ax.scatter(gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z, c='grey', alpha=.2)
                
            if show_diff:
                dif_gt_nn_sc = (rot_gt[current_rot]>0).astype(int)-(rot_arr_RA_sc[current_rot]>0).astype(int)
                
                gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z = np.nonzero(dif_gt_nn_sc<0)
                ax.scatter(gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z, c='blue')
                
                gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z = np.nonzero(dif_gt_nn_sc>0)
                ax.scatter(gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z, c='red')
    for current_rot in range(len(rot_gt)):      
        if 'nnUNet' in model_used:
            cntr += 1
            ax = fig.add_subplot(nbr_rot, nbr_model,cntr, projection='3d')
            angle_ = f'nnUNet {angle_title[current_rot][0]}z {angle_title[current_rot][1]}y {angle_title[current_rot][2]}z'
            ax.set_title(angle_)
            
            if show_softmax:
                nn_scs_val = rot_arr_RA_RA_m[current_rot] * (rot_arr_RA_RA[current_rot]>0).astype(int)
                gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z = np.nonzero(nn_scs_val)
                cs = nn_scs_val[gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z]
                c=scalarMap.to_rgba(cs)
                ax.scatter(gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z, c=c)
            else:
                gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z = np.nonzero(rot_arr_RA_RA[current_rot])
                ax.scatter(gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z, c='grey', alpha=.2)
                
            if show_diff:
                dif_gt_nn_sc = (rot_gt[current_rot]>0).astype(int)-(rot_arr_RA_RA[current_rot]>0).astype(int)
                
                gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z = np.nonzero(dif_gt_nn_sc<0)
                ax.scatter(gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z, c='blue')
                
                gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z = np.nonzero(dif_gt_nn_sc>0)
                ax.scatter(gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z, c='red')
    for current_rot in range(len(rot_gt)):
        if 'Bispectral' in model_used:
            cntr += 1
            ax = fig.add_subplot(nbr_rot, nbr_model,cntr, projection='3d')
            angle_ = f'Bispectral {angle_title[current_rot][0]}z {angle_title[current_rot][1]}y {angle_title[current_rot][2]}z'
            ax.set_title(angle_)
            
            if show_softmax:
                nn_scs_val = rot_arr_RA_bi_m[current_rot] * (rot_arr_RA_bi[current_rot]>0).astype(int)
                gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z = np.nonzero(nn_scs_val)
                cs = nn_scs_val[gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z]
                c=scalarMap.to_rgba(cs)
                ax.scatter(gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z, c=c)
            else:
                gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z = np.nonzero(rot_arr_RA_bi[current_rot])
                ax.scatter(gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z, c='grey', alpha=.2)
                
            if show_diff:
                dif_gt_nn_sc = (rot_gt[current_rot]>0).astype(int)-(rot_arr_RA_bi[current_rot]>0).astype(int)
                
                gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z = np.nonzero(dif_gt_nn_sc<0)
                ax.scatter(gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z, c='blue')
                
                gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z = np.nonzero(dif_gt_nn_sc>0)
                ax.scatter(gt_nn_sc_x, gt_nn_sc_y, gt_nn_sc_z, c='red')
                


def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot_cube(cube, coor = None, angle=320,print=False, expand=False):

    cube = normalize(cube)
    
    facecolors = cm.viridis(cube)
    #facecolors = cm.Greys(cube)
    facecolors[:,:,:,-1] = cube
    if coor is not None:
        facecolors[coor] = [1,0,0,1]
    if expand: 
        facecolors = explode(facecolors)
    filled = facecolors[:,:,:,-1] != 0
    if expand:
        x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    else:
        x, y, z = np.indices(np.array(filled.shape))
    if print: 
        fig = plt.figure(figsize=(30/2.54, 30/2.54))
        ax = fig.add_subplot(projection = '3d')
        #ax.view_init(30, angle)
        
        ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
        plt.show()
    return facecolors, x,y,z, filled

def compute_metrics(path_metrics_json, M=24, TTA = False):

    metric_names = []
    with open(path_metrics_json) as f:
        data = json.load(f)

    data_to_return = []
    rotations = []
    if TTA:    
        tmp_rot_c1, tmp_rot_c2, tmp_rot_c0 = [],[],[]
        for i in range(len(data)):
            tmp_c1, tmp_c2, tmp_c0 = [],[],[]
            for k in data[i].keys():
                print(k)
                rotations.append(k)
                if k == 'TFP':
                    for c in data[i][k].keys():
                        recall = data[i][k][c]['TP']/(data[i][k][c]['TP']+data[i][k][c]['FN'])
                        if (data[i][k][c]['TP']+data[i][k][c]['FP']):
                            precision = data[i][k][c]['TP']/(data[i][k][c]['TP']+data[i][k][c]['FP'])
                        else:
                            precision = 1.
                        accuracy = (data[i][k][c]['TP']+data[i][k][c]['TN'])/(data[i][k][c]['TP']+data[i][k][c]['FP']+data[i][k][c]['FN']+data[i][k][c]['TN'])
                        if int(c):
                            if int(c) - 1:
                                tmp_c2 += [recall, precision, accuracy]
                            else:
                                tmp_c1 += [recall, precision, accuracy]
                        else:
                            tmp_c0 += [recall, precision, accuracy]
                    if not i and not rot_id: metric_names += ['recall', 'precision', 'accuracy']
                else:
                    tmp_c0.append(data[i][k]['0'])
                    tmp_c1.append(data[i][k]['1'])
                    tmp_c2.append(data[i][k]['2'])
                    if not i and not rot_id: metric_names.append(k)
            tmp_rot_c0.append(tmp_c0)
            tmp_rot_c1.append(tmp_c1)
            tmp_rot_c2.append(tmp_c2)
        #metric_names = list(data[i].keys())[:-1] + ['recall', 'precision', 'accuracy']
        data_to_return.append([tmp_rot_c0, tmp_rot_c1, tmp_rot_c2])
    else:
        for i in range(0,len(data), M):
            # no rotation i, i+1..M rotation
            tmp_rot_c1, tmp_rot_c2, tmp_rot_c0 = [],[],[]
            for rot_id in range(M):
                for k in data[i+rot_id].keys():
                    rotations.append(k)
                    tmp_c1, tmp_c2, tmp_c0 = [],[],[]
                    #print(f'k {data[i+rot_id].keys()} met {data[i+rot_id][k].keys()}')
                    for met in data[i+rot_id][k].keys():
                        if met == 'TFP':
                            for c in data[i+rot_id][k][met].keys():
                                recall = data[i+rot_id][k][met][c]['TP']/(data[i+rot_id][k][met][c]['TP']+data[i+rot_id][k][met][c]['FN'])
                                if (data[i+rot_id][k][met][c]['TP']+data[i+rot_id][k][met][c]['FP']):
                                    precision = data[i+rot_id][k][met][c]['TP']/(data[i+rot_id][k][met][c]['TP']+data[i+rot_id][k][met][c]['FP'])
                                else:
                                    precision = 1.
                                accuracy = (data[i+rot_id][k][met][c]['TP']+data[i+rot_id][k][met][c]['TN'])/(data[i+rot_id][k][met][c]['TP']+data[i+rot_id][k][met][c]['FP']+data[i+rot_id][k][met][c]['FN']+data[i+rot_id][k][met][c]['TN'])
                                if int(c):
                                    if int(c) - 1:
                                        tmp_c2 += [recall, precision, accuracy]
                                    else:
                                        tmp_c1 += [recall, precision, accuracy]
                                else:
                                    tmp_c0 += [recall, precision, accuracy]
                                    
                            if not i and not rot_id: metric_names += ['recall', 'precision', 'accuracy']
                        else:
                            tmp_c0.append(data[i+rot_id][k][met]['0'])
                            tmp_c1.append(data[i+rot_id][k][met]['1'])
                            tmp_c2.append(data[i+rot_id][k][met]['2'])
                            
                            
                        
                            if not i and not rot_id: metric_names.append(met)
                    tmp_rot_c0.append(tmp_c0)
                    tmp_rot_c1.append(tmp_c1)
                    tmp_rot_c2.append(tmp_c2)
            data_to_return.append([tmp_rot_c0, tmp_rot_c1, tmp_rot_c2])
            #metric_names = list(data[i+rot_id][k].keys())[:-1] + ['recall', 'precision', 'accuracy']
    

        

    data_to_return = np.stack(data_to_return)
    
    return data_to_return, metric_names, np.unique(rotations)

def show_metrics(all_data,
                 metric_names, 
                 rotations,
                 not_used_metrics=[], 
                 colors = ['r', 'blue', 'purple', 'orange', 'black','gray','green'],
                 box_plot = True,
                 rot_wise = False,
                 return_dev = False,
                 return_out = False,
                 return_mean = False,
                 aggregate = None,
                 show_outliers = True,
                 label_angle=0,
                 eps = 0.001,
                 title = '24 right angles rotations comparison'
                 ):

    nbr_model = len(all_data)

    for model_id in range(nbr_model):

        if aggregate is None:
            all_data[model_id][0] =  np.moveaxis(all_data[model_id][0],[0,2],[1,-1])

        elif aggregate.lower() == 'mean_per_rot':
            all_data[model_id][0] =  np.mean(all_data[model_id][0], axis=2)

        elif aggregate.lower() == 'mean_per_image':
            all_data[model_id][0] = np.mean(all_data[model_id][0], axis=0)


    if len(all_data[0][0].shape) > 3:
        # 3 52 6 24
        m_index = -2
    else:
        m_index = -1
    nbr_rows = all_data[0][0].shape[m_index] - len(not_used_metrics)
    if box_plot is not None and rot_wise:
        nbr_class = 2 * nbr_model
    else:
        nbr_class = 2

    c1,c2 =[],[]
    
    
    print_deviation = []
    print_mean = []

    x_ticks = [i[1] for i in all_data] if box_plot is not None and not rot_wise else rotations
    
    if 'TTA' in all_data[0][1] and box_plot is None:
        x_ticks = [i+1 for i in range(all_data[0][0].shape[1])]
        
    x = [i+1 if box_plot is not None else i for i in range(len(x_ticks))]

    fig, ax = plt.subplots(nbr_rows, nbr_class, figsize=[nbr_class*13, nbr_rows*5])
    if box_plot is not None and not rot_wise:
        top = .9
        title = title + '\n '
        for i in range(nbr_model):
            title += f'{i+1}: {all_data[i][1]}'
            if nbr_model - i > 1:
                title += ', '
            if not (i+1)%2:
                title += '\n'
                top -= 0.025
        plt.subplots_adjust(top=top)
            
    fig.suptitle(title, fontsize=15)
    skip_metric = 0
    y_limits = []
    box_dict = []
    label_metrics = metric_names.copy()
                
    for metric in range(all_data[0][0].shape[m_index]):
        if metric_names[metric] in not_used_metrics:
            skip_metric += 1
            label_metrics.remove(metric_names[metric])
            continue
        #if 'TTA' in all_data[0][1]:
        print(f'Metric {label_metrics[metric-skip_metric]}')
        for id, model_data in enumerate(all_data):
            
            # None, boxplot, violinplot
            if box_plot is not None and rot_wise:
                if len(all_data[0][0].shape) > 3:
                    
                    if box_plot == 'boxplot':
                        box_dict.append(ax[metric-skip_metric,id].boxplot(model_data[0][1,:,metric], showfliers=show_outliers))
                        box_dict.append(ax[metric-skip_metric,nbr_model+id].boxplot(model_data[0][2,:,metric], showfliers=show_outliers))
                    elif box_plot == 'violinplot':
                        box_dict.append(ax[metric-skip_metric,id].violinplot(model_data[0][1,:,metric]))
                        box_dict.append(ax[metric-skip_metric,nbr_model+id].violinplot(model_data[0][2,:,metric]))
                else:
                    c1.append(model_data[0][1,:,metric])
                    c2.append(model_data[0][2,:,metric])
                limit_metric = [model_data[0][1,:,metric], model_data[0][2,:,metric]]
                    
            elif box_plot is not None:
                if len(all_data[0][0].shape) > 3:
                    # 3 52 6 24
                    # 3 6 52 24
                    # 52*24 6
                    limit_metric = [np.reshape(np.moveaxis(model_data[0][1],1,2), [-1,all_data[0][0].shape[m_index]])[:,metric], 
                                    np.reshape(np.moveaxis(model_data[0][2],1,2), [-1,all_data[0][0].shape[m_index]])[:,metric]]
                    
                    aa = np.reshape(np.moveaxis(model_data[0][0],1,2), [-1,all_data[0][0].shape[m_index]])[:,metric]
                    if label_metrics[metric-skip_metric] == 'RMSE':
                        print(f'Model {model_data[1]}:\n Mean c1 {np.mean(limit_metric[0])*10000:.2f}e-4 c2 {np.mean(limit_metric[1])*10000:.2f}e-4')
                    elif label_metrics[metric-skip_metric] == 'HD':
                        print(f'Model {model_data[1]}:\n Mean c1 {np.mean(limit_metric[0]):.2f} c2 {np.mean(limit_metric[1]):.2f}')
                    else:
                        print(f'Model {model_data[1]}:\n Mean c1 {np.mean(limit_metric[0])*100:.2f}% c2 {np.mean(limit_metric[1])*100:.2f}%')
                    c1.append(limit_metric[0])
                    c2.append(limit_metric[1]) 
                    
                else:
                    c1.append(model_data[0][1,:,metric])
                    c2.append(model_data[0][2,:,metric])
                    limit_metric = [model_data[0][1,:,metric], model_data[0][2,:,metric]]
                
            else:
                ax[metric-skip_metric,0].plot(model_data[0][1,:,metric],label=f'{model_data[1]}', color=colors[id])
                ax[metric-skip_metric,1].plot(model_data[0][2,:,metric],label=f'{model_data[1]}', color=colors[id])
                c1.append(model_data[0][1,:,metric])
                c2.append(model_data[0][2,:,metric])
                limit_metric = [model_data[0][1,:,metric], model_data[0][2,:,metric]]
                
                #if 'TTA' in all_data[0][1]:
                c_mean = [np.mean(model_data[0][1,:,metric]),np.mean(model_data[0][2,:,metric])]
                #ax[metric-skip_metric,0].plot([c_mean[0] for _ in range(model_data[0][1,:,metric].shape[0])],label=f'{model_data[1]} mean', color='light'+colors[id])
                #ax[metric-skip_metric,1].plot([c_mean[1] for _ in range(model_data[0][1,:,metric].shape[0])],label=f'{model_data[1]} mean', color='light'+colors[id])
                if label_metrics[metric-skip_metric] == 'RMSE':
                    print(f'Model {model_data[1]}:\n Mean c1 {c_mean[0]*10000:.2f}e-4 c2 {c_mean[1]*10000:.2f}e-4')
                elif label_metrics[metric-skip_metric] == 'HD':
                        print(f'Model {model_data[1]}:\n Mean c1 {np.mean(limit_metric[0]):.2f} c2 {np.mean(limit_metric[1]):.2f}')
                else:
                    print(f'Model {model_data[1]}:\n Mean c1 {c_mean[0]*100:.2f}% c2 {c_mean[1]*100:.2f}%')
            if id:
                if np.max(limit_metric[0]) >  y_limits[-1][0]:
                    y_limits[-1][0] = np.max(limit_metric[0])
                    
                if np.max(limit_metric[1]) >  y_limits[-1][0]:
                    y_limits[-1][0] = np.max(limit_metric[1])
                    
                if np.min(limit_metric[0]) <  y_limits[-1][1]:
                    y_limits[-1][1] = np.min(limit_metric[0])
                    
                if np.min(limit_metric[1]) <  y_limits[-1][1]:
                    y_limits[-1][1] = np.min(limit_metric[1])
            else:
                if np.max(model_data[0][1,:,metric]) > np.max(model_data[0][2,:,metric]):
                    max_zero =  np.max(model_data[0][1,:,metric])
                else:
                    max_zero =  np.max(model_data[0][2,:,metric])
                    
                if np.min(model_data[0][1,:,metric]) > np.min(model_data[0][2,:,metric]):
                    min_zero =  np.min(model_data[0][2,:,metric])
                else:
                    min_zero =  np.min(model_data[0][1,:,metric])
                    
                y_limits.append([max_zero, min_zero])

        print_deviation.append([c1,c2,label_metrics[metric-skip_metric]])

        if box_plot == 'boxplot':
            box_dict.append(ax[metric-skip_metric,0].boxplot(c1, showfliers=show_outliers))
            box_dict.append(ax[metric-skip_metric,1].boxplot(c2, showfliers=show_outliers))
        elif box_plot == 'violinplot':
            box_dict.append(ax[metric-skip_metric,0].violinplot(c1))
            box_dict.append(ax[metric-skip_metric,1].violinplot(c2))
            
        c1, c2 = [],[]

    label_metrics = metric_names.copy()
    for i in range(nbr_class):
        if box_plot is not None and rot_wise:
            ax[0,i%int(nbr_class/nbr_model) * nbr_model + i//int(nbr_class/nbr_model)].set_title(f"{all_data[i//int(nbr_class/nbr_model)][1]} Class {i%int(nbr_class/nbr_model)+1}")
        else:
            ax[0,i].set_title(f"Class {i+1}")
        j_skip = 0
        for j in range(all_data[0][0].shape[m_index]):
            if metric_names[j] in not_used_metrics:
                j_skip += 1
                if metric_names[j] in label_metrics:
                    label_metrics.remove(metric_names[j])
                continue
                
            if box_plot is not None:
                if not rot_wise:
                    pass
                else:
                    ax[j-j_skip,i].set_xticks(x, [] , rotation=label_angle)
                ax[j-j_skip, i].set_ylim([y_limits[j-j_skip][1]-eps, y_limits[j-j_skip][0]+eps])

            else: 
                ax[j-j_skip,i].set_xticks(x, [] , rotation=label_angle)
            ax[j-j_skip,i].set_ylabel(f'{label_metrics[j-j_skip]}')
            if box_plot is None:
                ax[j-j_skip,i].legend()
            ax[j-j_skip,i].grid(alpha=0.5)
        if box_plot is not None and not rot_wise:
            pass
        else:
            ax[nbr_rows-1,i].set_xticks(x, x_ticks , rotation=label_angle)
        ax[nbr_rows-1,i].set_ylabel(f'{label_metrics[j-j_skip]}')
        if box_plot is None:
            ax[nbr_rows-1,i].legend()
        ax[nbr_rows-1,i].grid(alpha=0.5)
    if return_dev and return_out:
        return print_deviation, box_dict
    elif return_dev:
        return print_deviation
    elif return_out:
        return box_dict

def do_plots(all_data, all_data_folds, mean_fold, only_mean, show_boxplot, metric_names, rotations, not_used_metrics, per_rotation, show_outliers, aggregate, title = '24 right angles rotations comparison', colors = ['r', 'green', 'blue', 'salmon', 'lightgreen', 'cyan']):
    
    if mean_fold:
        for m in range(len(all_data_folds)):
            tmp = []
            for f in all_data_folds[m]:
                #print(f[1])
                tmp.append(f[0])
            tmp_stack = np.stack(tmp, axis=-1)
            
            mean_val = np.mean(tmp_stack, axis=-1)
            all_data.append([mean_val, f[1][:-3]+' mean'])
            if not only_mean:
                max_val = np.max(np.stack(tmp, axis=-1), axis=-1)
                min_val = np.min(np.stack(tmp, axis=-1), axis=-1)
                all_data.append([max_val, f[1][:-3]+' max'])
                all_data.append([min_val, f[1][:-3]+' min'])

    if show_outliers and show_boxplot is not None:
        deviation = show_metrics(all_data,
                                metric_names=metric_names, 
                                rotations=rotations, 
                                not_used_metrics=not_used_metrics, 
                                colors = colors,
                                box_plot = show_boxplot, 
                                return_dev=True, 
                                return_out=False,
                                rot_wise=per_rotation,
                                aggregate=aggregate, 
                                show_outliers=show_outliers,
                                label_angle=90,
                                title=title
                                )
    else:
        deviation = show_metrics(all_data,
                                metric_names=metric_names, 
                                rotations=rotations, 
                                not_used_metrics=not_used_metrics, 
                                colors = colors,
                                box_plot = show_boxplot, 
                                return_dev=True, 
                                return_out=False,
                                rot_wise=per_rotation,
                                aggregate=aggregate, 
                                show_outliers=show_outliers,
                                label_angle=90,
                                title=title
                                )
    return deviation


def change_color(a, color_list, edge_col='black'):
    for partname in list(a.keys()):
        if partname == 'bodies':
            continue
        vp = a[partname]
        vp.set_edgecolor(edge_col)
        vp.set_linewidth(1)

    # Make the violin body blue with a red border:
    for vp in a['bodies']:
        vp.set_facecolor(color_list)
        vp.set_edgecolor(edge_col)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
        
def create_violin_subplot(ax, perfs, regs, model_names, x, y, y_name, rmse_prec = 4, color_list = ['blue','red','purple','yellow']):
    ax2 = ax.twinx()
    ax2.set_ylabel("RMSE")
    ax.set_ylabel("DICE")
    ax.grid()
    ax3 = ax.twiny()
    ax3.sharey(ax)
    if y_name is not None:
        ax3.set_xticks(y, y_name)
    edge_col = 'black'
    
    ax2.vlines((int(len(x)/2)+int(len(x)/2-1))/2+1,0,np.max(np.array(regs)[...,0]), colors='black')
    for k in range(0,2*len(perfs),2):
        j1 = int(k//2)
        j2 = int(k//2)
        a = ax.violinplot(perfs[int(j1)][...,1].flatten(), [j2+1])
        change_color(a, color_list[j1])
        model_names[j1] = model_names[j1] + f' Mean: {np.mean(perfs[int(j1)][...,1].flatten())*100:.2f}%'
        
        a = ax2.violinplot(regs[int(j1)][...,0].flatten(),[j2+len(perfs)+1])
        change_color(a, color_list[j1])
        model_names[j1+len(perfs)] = model_names[j1+len(perfs)] + f' Mean: {np.mean(regs[int(j1)][...,0].flatten())*(10**rmse_prec):.2f}e-{str(rmse_prec)}'
           
    ax.set_xticks(x, model_names)
    

def compute_t_student(dist_A, dist_B,p=0.05, wilcoxon=True, student=True):
    diff =(dist_A - dist_B)
    dist_mean = np.mean(diff)
    dist_std = np.std(diff)
    N = dist_A.shape[0] 
    t = np.sqrt(N)*dist_mean/dist_std

    high_p = 1-p/2
    t_val = stats.t.ppf(high_p, N-1)

    p_out = 2*(1 - stats.t.cdf(abs(t), N-1))
    if student:
        if p_out > p:
            print(f"  Paired t Fail to reject null hypotesis p:{p_out*100:.6f}%")
        else:
            print(f"  Paired t Reject null hypotesis p:{p_out*100:.6f}%")
        if not wilcoxon:
            print("\n")
    if wilcoxon:
        if (stats.wilcoxon(diff).pvalue) > p:
            print(f"  Wilcoxon Fail to reject null hypotesis p:{stats.wilcoxon(diff).pvalue*100:.4f}%\n")
        else:
            print(f"  Wilcoxon Reject null hypotesis p:{stats.wilcoxon(diff).pvalue*100:.4f}%\n")
        

    return t, t_val,  p_out

def create_violin_subplot_2(ax, perfs, regs, model_names, x, y, y_name, rmse_prec = 4, color_list = ['blue','purple','green','dimgray'], color_list_rmse = ['lightblue','violet','lightgreen','lightgray'], show_means=True):
    ax2 = ax.twinx()
    ax2.set_ylabel("RMSE")
    ax.set_ylabel("DICE")
    ax.grid()
    showmeans = False
    showextrema = False
    edge_col = 'black'
    
    
    for k in range(0,2*len(perfs),2):
        j1 = int(k//2)
        j2 = int(k//2)
        
        eps=0.1
                
        lim_reg = (np.min(regs[int(j1)][...,0]), np.max(regs[int(j1)][...,0]))
        lim_perf =(np.min(perfs[int(j1)][...,1]), np.max(perfs[int(j1)][...,1])) 
            
        ax.vlines(j1+1,lim_perf[0],lim_perf[1], colors=color_list[j1], linewidth=1)
        ax.hlines(lim_perf[1], (j1+1)-eps,j1+1, colors=color_list[j1], linewidth=1)
        ax.hlines(lim_perf[0], (j1+1)-eps,j1+1, colors=color_list[j1], linewidth=1)
        
        
        ax2.vlines(j1+1,lim_reg[0],lim_reg[1], colors=color_list_rmse[j1], linewidth=1)
        ax2.hlines(lim_reg[1], j1+1,j1+1+eps, colors=color_list_rmse[j1], linewidth=1)
        ax2.hlines(lim_reg[0], j1+1,j1+1+eps, colors=color_list_rmse[j1], linewidth=1)
        
        a = ax.violinplot(perfs[int(j1)][...,1].flatten(), [j2+1],showmeans=showmeans, showextrema=showextrema, showmedians=False)
        change_color(a, color_list[j1])
        if show_means:
            model_names[j1] = model_names[j1] + f'\n Mean DSC: {np.mean(perfs[int(j1)][...,1].flatten())*100:.2f}%'#\nVar DSC: {np.var(perfs[int(j1)][...,1].flatten())*100:.2f}%'
        
        eps = 0.005
        for b in a['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further right than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)

        a2 = ax2.violinplot(regs[int(j1)][...,0].flatten(),[j2+1], showmeans=showmeans, showextrema=showextrema, showmedians=False)
        
        change_color(a2, color_list_rmse[j1])
        if show_means:
            model_names[j1] = model_names[j1] + f'\n Mean RMSE: {np.mean(regs[int(j1)][...,0].flatten())*(10**rmse_prec):.2f}e-{str(rmse_prec)}'#\nVar RMSE: {np.var(regs[int(j1)][...,0].flatten())*(10**rmse_prec):.2f}e-{str(rmse_prec)}'
        
        for b in a2['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further left than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)

        ax.yaxis.label.set_color(color_list[j1])
        ax.tick_params(axis='y', colors=color_list[j1])
        ax2.yaxis.label.set_color(color_list_rmse[j1])
        ax2.tick_params(axis='y', colors=color_list_rmse[j1])
        

    ax.set_xticks(x, model_names)
    
        
