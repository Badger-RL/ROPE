import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

import pdb
from matplotlib import rcParams
import os
import itertools

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def bool_argument(parser, name, default=False, msg=''):
    dest = name.replace('-', '_')
    parser.add_argument('--%s' % name, dest=dest, type=bool, default=default, help=msg)
    parser.add_argument('--no-%s' % name, dest=dest, type=bool, default=default, help=msg)

parser = argparse.ArgumentParser()
parser.add_argument('result_directory', help=help)
parser.add_argument('--plot_type', type = str)
parser.add_argument('--tr_metric', type = str)
parser.add_argument('--domain_name', type = str)
parser.add_argument('--batch', type = int, default = 1000)
parser.add_argument('--traj', type = int, default = 1000)
parser.add_argument('--method', type = str)
parser.add_argument('--plotting_stat', type = str)
parser.add_argument('--save_fname', type = str, default = None)
parser.add_argument('--y_label', type = str, default = '(relative) Abs. Err.')
parser.add_argument('--y_log', type = str2bool, default = False)

FLAGS = parser.parse_args()

def compute_stats(method, errors, plotting_stat = 'abs', print_log = False):

    errors = np.array(errors)
    n = len(errors) # trials

    if errors.ndim == 2:
        if plotting_stat == 'iqm_abs':
            errors_sorted = np.sort(errors, axis = 0)
            n = errors.shape[0]
            errors = errors_sorted[int(np.floor(n/4)):int(np.ceil(3*n/4)), :]
            n = errors.shape[0]
            mean = np.mean(errors, axis = 0)
            std = np.std(errors, axis = 0)
        else:
            mean = np.mean(errors, axis = 0)
            std = np.std(errors, axis = 0)
    else:
        if plotting_stat == 'mse':
            mean = np.mean(np.square(errors))
            std = np.std(np.square(errors))
        elif plotting_stat == 'abs':
            n = len(errors)
            mean = np.mean(errors)
            std = np.std(errors)
        elif plotting_stat == 'iqm_abs':
            # IQM
            vals_sorted = np.sort(errors)
            errors = vals_sorted[int(np.floor(n/4)):int(np.ceil(3*n/4))]
            n = len(errors)
            mean = np.mean(errors)
            std = np.std(errors)
        
    yerr = 1.96 * std / np.sqrt(float(n))
    ylower = mean - yerr
    yupper = mean + yerr

    stats = {
        'mean': mean,
        'yerr': yerr,
        'ylower': ylower,
        'yupper': yupper
    }
    if print_log and errors.ndim == 1:
        print ('num trials for {}: {}, mean {}, ylower {}, yupper {}'.format(method, n, mean, ylower, yupper))
    return stats

def _linestyle_color_combo(num_hp):
    ls = [':', '-', '-.']
    colors = ['#4C72B0', '#C44E52', '#55A868']
    combined = [ls, colors]
    combined = sorted(list(itertools.product(*combined)))
    return combined[:num_hp]

def plot_training_all(data, batch, traj, file_name, plot_params, metric = 'err'):
    for data_name in data:
        for method in data[data_name]:
            fig, ax = plt.subplots()
            fig.set_size_inches(13.5, 12.0, forward=True)
            # if 'off-policy-sa' in method:
            #     method = 'ROPE'
            # elif 'identity' in method:
            #     method = 'FQE'
            # elif 'target-phi-sa' in method:
            #     method = 'pie-critic'
            # else:
            #     continue
            print ('method: {}'.format(method))

            num_hps = len(data[data_name][method][batch][traj])
            ls_colors_combo = _linestyle_color_combo(num_hps)
            for idx, hp in enumerate(sorted(data[data_name][method][batch][traj])):
                if metric not in data[data_name][method][batch][traj][hp] or len(data[data_name][method][batch][traj][hp][metric]) == 0:
                    continue
                sub_data = np.array(data[data_name][method][batch][traj][hp][metric])

                if method == 'fqe_off-policy-sa':
                    label = (hp[-2], hp[-1])
                else:
                    label = (hp[0])
                ls = ls_colors_combo[idx][0]
                color = ls_colors_combo[idx][1]
                stats = compute_stats(method, sub_data, print_log = True, plotting_stat = FLAGS.plotting_stat)
                y = stats['mean']
                ylower = stats['ylower']
                yupper = stats['yupper']

                tr_steps = sub_data.shape[1]
                #x = list(sub_data[0].keys())#[i * 1000 for i in range(1, tr_steps + 1)]       
                x = [i * 1000 for i in range(1, tr_steps + 1)]       

                x = np.array(x)
                y = np.array(y)
                ylower = np.array(ylower)
                yupper = np.array(yupper)
                #x = np.arange(len(y))
                line, = plt.plot(x, y, label=label, linestyle = ls, linewidth = 8, color = color)
                #line, = plt.plot(x, y, label=label)
                color = line.get_color()
                alpha = 0.5
                plt.fill_between(x, ylower, yupper, facecolor=color, alpha=alpha)

            if plot_params['log_scale']:
                #ax.set_xscale('log')
                ax.set_yscale('log')
            if plot_params['x_range'] is not None:
                plt.xlim(plot_params['x_range'])
            if plot_params['y_range'] is not None:
                plt.ylim(plot_params['y_range'])

            ax.xaxis.set_tick_params(labelsize=plot_params['tfont'])
            ax.yaxis.set_tick_params(labelsize=plot_params['tfont'])

            ax.set_xlabel(plot_params['x_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])
            ax.set_ylabel(plot_params['y_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])

            if plot_params['legend']:
                plt.legend(fontsize=plot_params['lfont'], loc=plot_params['legend_loc'],
                        ncol=plot_params['legend_cols'])
            fig.tight_layout()
            if FLAGS.save_fname:
                plt.savefig('{}_{}_{}.jpg'.format(FLAGS.save_fname, method, metric))
            else:
                plt.savefig('{}_{}.jpg'.format(file_name, data_name))
            plt.close()
        #plt.show()

def plot_training(data, batch, traj, file_name, plot_params, metric = 'err'):
    for data_name in data:
        fig, ax = plt.subplots()
        fig.set_size_inches(13.5, 12.0, forward=True)
        for method in data[data_name]:
            linestyle = '-'
            if 'off-policy-sa' in method:
                label = 'ROPE'
                linestyle = '-'
                color = '#4C72B0'
            elif 'identity' in method:
                if ('clip' not in method) and ('deep' not in method):
                    label = 'FQE'
                    linestyle = '-.'
                    color = '#C44E52'
                elif 'clip' in method:
                    label = 'FQE-CLIP'
                    linestyle = ':'
                    color = '#55A868'
                elif 'deep' in method:
                    continue

            elif 'target-phi-sa' in method:
                label = '$\pi_e$-critic'
                linestyle = ':'
                color = '#55A868'
            else:
                continue
            print ('method: {}'.format(method))
            if metric == 'err':
                sub_data = np.array(data[data_name][method][batch][traj]['errs_tr'])
            else:
                sub_data = np.array(data[data_name][method][batch][traj]['r_ests'])

            stats = compute_stats(method, sub_data, print_log = True, plotting_stat = FLAGS.plotting_stat)
            y = stats['mean']
            ylower = stats['ylower']
            yupper = stats['yupper']

            tr_steps = sub_data.shape[1]
            #x = list(sub_data[0].keys())#[i * 1000 for i in range(1, tr_steps + 1)]       
            x = [i * 1000 for i in range(1, tr_steps + 1)]       

            x = np.array(x)
            y = np.array(y)
            ylower = np.array(ylower)
            yupper = np.array(yupper)
            #x = np.arange(len(y))
            line, = plt.plot(x, y, label=label, linestyle = linestyle, linewidth = 8, color = color)
            #line, = plt.plot(x, y, label=label)
            color = line.get_color()
            alpha = 0.5
            plt.fill_between(x, ylower, yupper, facecolor=color, alpha=alpha)
        
        if plot_params['log_scale']:
            #ax.set_xscale('log')
            ax.set_yscale('log')
        if plot_params['x_range'] is not None:
            plt.xlim(plot_params['x_range'])
        if plot_params['y_range'] is not None:
            plt.ylim(plot_params['y_range'])

        ax.xaxis.set_tick_params(labelsize=plot_params['tfont'])
        ax.yaxis.set_tick_params(labelsize=plot_params['tfont'])

        ax.set_xlabel(plot_params['x_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])
        ax.set_ylabel(plot_params['y_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])

        if plot_params['legend']:
            plt.legend(fontsize=plot_params['lfont'], loc=plot_params['legend_loc'],
                       ncol=plot_params['legend_cols'])
        fig.tight_layout()
        if FLAGS.save_fname:
            plt.savefig('{}.jpg'.format(FLAGS.save_fname))
        else:
            plt.savefig('{}_{}.jpg'.format(file_name, data_name))
        plt.close()
        #plt.show()

def plot_bar_hp(data_, batch, traj, file_name, plot_params, metric = 'err', hp = False, hp_name = 'phi_beta'):
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 15.0, forward=True)

    main_results_to_plot = {}
    for env in data_:
        for method in data_[env]:
            print (method)
            if ('fqe_identity' not in method) and method != 'fqe_off-policy-sa' and method != 'fqe_bcrl':
                continue
            
            if method not in main_results_to_plot:
                main_results_to_plot[method] = {}
            
            for hp in data_[env][method][batch][traj]:
                vals = data_[env][method][batch][traj][hp][metric]
                stats = compute_stats(method + str(hp), vals, print_log = True, plotting_stat = FLAGS.plotting_stat)
                main_results_to_plot[method][hp] = stats
        
        methods = ['fqe_identity', 'fqe_off-policy-sa']

        for method in methods:
            if method not in main_results_to_plot:
                continue
            for hp in sorted(main_results_to_plot[method]):
                stats = main_results_to_plot[method][hp]
                if method == 'fqe_off-policy-sa':
                    print (stats['mean'])
                    plt.bar(str('{}\n{}'.format(hp[-2], hp[-1])), stats['mean'], yerr = stats['yerr'])
                else:
                    plt.bar(str('FQE'), stats['mean'], yerr = stats['yerr'])
    
        if plot_params['log_scale']:
            ax.set_yscale('log')
        if plot_params['x_range'] is not None:
            plt.xlim(plot_params['x_range'])
        if plot_params['y_range'] is not None:
            plt.ylim(plot_params['y_range'])

        ax.xaxis.set_tick_params(labelsize=plot_params['tfont'])
        ax.yaxis.set_tick_params(labelsize=plot_params['tfont'])

        ax.set_xlabel(plot_params['x_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])
        ax.set_ylabel(plot_params['y_label'], fontsize=plot_params['bfont'], labelpad = plot_params['axis_label_pad'])

        # if plot_params['legend']:
        #     plt.legend(fontsize=plot_params['lfont'], loc=plot_params['legend_loc'],
        #             ncol=plot_params['legend_cols'])
        fig.tight_layout()
        print (file_name)
        if FLAGS.save_fname:
            plt.savefig('{}.jpg'.format(FLAGS.save_fname))
        else:
            plt.savefig('{}_{}.jpg'.format(file_name, data_name))
        plt.close()

def collect_data():

    data = {}

    for basename in os.listdir(FLAGS.result_directory):
        if '.npy' not in basename:
            continue
        f_name = os.path.join(FLAGS.result_directory, basename)
        try:
            results = np.load(f_name, allow_pickle = True).item()
        except Exception as e:
            if 'Tag' not in str(e):
                raise e
        # seed_574976_batch_50_traj_10_Qlr_0.001_Wlr_0.001_lamlr_0.001.npy
        '''
        method -> batch -> traj -> (qlr, wlr, lamlr) -> {e1, e2, ..., en}
        '''
        summary = np.load(f_name, allow_pickle = True).item()

        domain_name = summary['env']
        batch = 1000#summary['batch_size'] 
        traj_len = 1000#summary['traj_len']
        qlr =  summary['hp']['Q_lr']
        philr = summary['hp']['phi_lr'] if 'phi_lr' in summary['hp'] else 0
        phi_outdim = summary['hp']['phi_outdim'] if 'phi_outdim' in summary['hp'] else 0
        beta = summary['hp']['beta'] if 'beta' in summary['hp'] else 0
        hp = (qlr, philr, phi_outdim, beta)#, phi_rep_alpha)
        oracle_est = summary['oracle_est']
        rand_est = summary['rand_est']
        coverage = summary['coverage'] if 'coverage' in summary else 0
        results = summary['results']
        if 'HumanoidStandup' in list(summary['results'].keys())[0]:
            results['medium-expert'] = summary['results'].pop('HumanoidStandup_batch_100_mix-ratio_0.5')
        data_names = results.keys()

        for data_name in data_names:
            if data_name not in data:
                data[data_name] = {}
            for algo in summary['results'][data_name]:
                if algo not in data[data_name]:
                    data[data_name][algo] = {}
                if batch not in data[data_name][algo]:
                    data[data_name][algo][batch] = {}
                if traj_len not in data[data_name][algo][batch]:
                    data[data_name][algo][batch][traj_len] = {}
                if hp not in data[data_name][algo][batch][traj_len]:
                    data[data_name][algo][batch][traj_len][hp] = {
                        'err': [],
                        'errs_tr': [],
                        'oracle_est': [],
                        'r_ests': [],
                        'r_est': [],
                        'ope_tr_losses': [],
                        'phi_tr_losses': [],
                        'phi_mean_dim': [],
                        'phi_std_dim': []
                    }
                data[data_name][algo][batch][traj_len][hp]['err'].append(results[data_name][algo]['err'] if 'err' in results[data_name][algo] else 0)
                data[data_name][algo][batch][traj_len][hp]['errs_tr'].append(results[data_name][algo]['errs_tr'] if 'errs_tr' in results[data_name][algo] else 0)
                data[data_name][algo][batch][traj_len][hp]['r_est'].append(results[data_name][algo]['r_est'] if 'r_est' in results[data_name][algo] else 0)
                data[data_name][algo][batch][traj_len][hp]['r_ests'].append(results[data_name][algo]['r_ests'] if 'r_ests' in results[data_name][algo] else 0)
                data[data_name][algo][batch][traj_len][hp]['oracle_est'].append(oracle_est)
                data[data_name][algo][batch][traj_len][hp]['ope_tr_losses'].append(results[data_name][algo]['ope_tr_losses'] if 'ope_tr_losses' in results[data_name][algo] else 0)
                data[data_name][algo][batch][traj_len][hp]['phi_tr_losses'].append(results[data_name][algo]['phi_tr_losses'] if 'ope_tr_losses' in results[data_name][algo] else 0)
                data[data_name][algo][batch][traj_len][hp]['phi_mean_dim'].append(results[data_name][algo]['phi_mean_dim'] if 'phi_mean_dim' in results[data_name][algo] else 0)
                data[data_name][algo][batch][traj_len][hp]['phi_std_dim'].append(results[data_name][algo]['phi_std_dim'] if 'phi_std_dim' in results[data_name][algo] else 0)


    return data#, domain_name

def best_hp(data):
    best_data = {}
    for data_name in data:
        best_data[data_name] = {}
        for method in data[data_name]:
            best_data[data_name][method] = {}
            for batch in data[data_name][method]:
                best_data[data_name][method][batch] = {}
                for traj_len in data[data_name][method][batch]:
                    best_data[data_name][method][batch][traj_len] = {}
                    min_err = float('inf')
                    min_err_bound = float('inf')
                    best_coverage = -1
                    best_hp = -1
                    for hp in data[data_name][method][batch][traj_len]:
                        errs = np.array(data[data_name][method][batch][traj_len][hp]['err'])
                        num = len(errs)

                        stats = compute_stats(method, errs, plotting_stat = FLAGS.plotting_stat)
                        val = stats['mean']
                        if val < min_err:
                            min_err = val
                            best_hp = hp
                            min_err_bound = stats['yerr']
                        print ('dataset {} method {} batch {} traj {} best hp {}, trials: {}, err {}, bound: {}'.format(data_name, method, batch, traj_len, hp, num, val, stats['yerr']))
                    print ('BEST: dataset {} method {} batch {} traj {} best hp {}, err: {}'.format(data_name, method, batch, traj_len, best_hp, str(round(min_err, 3)) + " \pm " + str(round(min_err_bound, 3))))
                    if best_hp == -1:
                        best_hp = hp
                    
                    best_data[data_name][method][batch][traj_len]['err'] = data[data_name][method][batch][traj_len][best_hp]['err']
                    best_data[data_name][method][batch][traj_len]['errs_tr'] = data[data_name][method][batch][traj_len][best_hp]['errs_tr']
                    best_data[data_name][method][batch][traj_len]['r_est'] = data[data_name][method][batch][traj_len][best_hp]['r_est']
                    best_data[data_name][method][batch][traj_len]['r_ests'] = data[data_name][method][batch][traj_len][best_hp]['r_ests']
                    best_data[data_name][method][batch][traj_len]['oracle_est'] = data[data_name][method][batch][traj_len][best_hp]['oracle_est']
                    best_data[data_name][method][batch][traj_len]['ope_tr_losses'] = data[data_name][method][batch][traj_len][best_hp]['ope_tr_losses']
                    best_data[data_name][method][batch][traj_len]['phi_tr_losses'] = data[data_name][method][batch][traj_len][best_hp]['phi_tr_losses']
                    best_data[data_name][method][batch][traj_len]['phi_mean_dim'] = data[data_name][method][batch][traj_len][best_hp]['phi_mean_dim']
                    best_data[data_name][method][batch][traj_len]['phi_std_dim'] = data[data_name][method][batch][traj_len][best_hp]['phi_std_dim']
    return best_data 

def main():

    if not FLAGS.result_directory:
        print ('No result directory given. Exiting.')
        return

    data, domain_name = collect_data(), FLAGS.result_directory.split('_')[2]
    best_hp_data = best_hp(data)
    nice_fonts = {
        #"pgf.texsystem": "pdflatex",
        # Use LaTex to write all text
        #"text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 20,
        "font.size": 20,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 16,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
    #plt.figure()
    plt.style.use('seaborn')
    #mpl.rcParams['font.family'] = 'serif'
    #mpl.rcParams['text.usetex'] = True
    #mpl.rcParams.update(nice_fonts)
    #sns.set(rc = nice_fonts)

    plot_params = {'bfont': 45,
               'lfont': 45,
               'tfont': 45,
               'legend': True,
               'legend_loc': 0,
               'legend_cols': 2,
               'y_range': None,
               'x_range': None,
               'log_scale': True,
               #'y_label': r'(relative) MSE($\rho(\pi_e)$)',
               'y_label': FLAGS.y_label,
               #'y_label': '(relative) MSE',
               'shade_error': True,
               'x_mult': 1,
               'axis_label_pad': 15}

    fname = '{}_'.format(domain_name)

    if FLAGS.plot_type == 'training':
        plot_params['x_label'] = 'training steps'
        plot_params['log_scale'] = False
        plot_params['y_range'] = (0,1)
        fname += '{}_training_b_{}_t_{}'.format(FLAGS.tr_metric, FLAGS.batch, FLAGS.traj)
        plot_training(best_hp_data, FLAGS.batch, FLAGS.traj, fname, plot_params, metric = FLAGS.tr_metric)
    elif FLAGS.plot_type == 'final_bar_hp_abl':
        plot_params['x_label'] = 'ROPE Dim (top) / $\\beta$ (bottom)'
        plot_params['log_scale'] = False
        plot_params['y_range'] = (0,1)
        fname += '{}_bar_b_{}_t_{}'.format(FLAGS.tr_metric, FLAGS.batch, FLAGS.traj)
        plot_bar_hp(data, FLAGS.batch, FLAGS.traj, fname, plot_params, metric = FLAGS.tr_metric, hp = True)
    elif FLAGS.plot_type == 'training_all':
        plot_params['x_label'] = 'training steps'
        plot_params['log_scale'] = FLAGS.y_log
        fname += '{}_bar_b_{}_t_{}'.format(FLAGS.tr_metric, FLAGS.batch, FLAGS.traj)
        plot_training_all(data, FLAGS.batch, FLAGS.traj, fname, plot_params, metric = FLAGS.tr_metric)
    
if __name__ == '__main__':
    main()


