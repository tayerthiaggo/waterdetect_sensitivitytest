import os
import numpy as np
import xarray as xr
import rasterio
import rioxarray as rxr
from sklearn.metrics import matthews_corrcoef, f1_score, cohen_kappa_score
import waterdetect as wd
import pandas as pd
import time
from IPython.display import clear_output

class SensitivityTestWD():   
    def __init__(self, gt_path, target_path, ini_file, target_bands = ['Blue', 'Green', 'Red', 'Nir'], max_cluster = (3,9,1), reg = (0.01, 0.1, 0.01), 
                 acc_metric = 'mcc', rst_ext = '.tif', target_path_swir = None, target_bands_swir = ['Mir', 'Mir2'], trigger = 1):
        
        self.gt_path = gt_path
        self.target_path = target_path
        self.ini_file = ini_file
        self.target_bands = target_bands
        self.max_cluster = max_cluster
        self.reg = reg
        self.acc_metric = acc_metric
        self.rst_ext = rst_ext
        self.target_path_swir = target_path_swir
        self.target_bands_swir = target_bands_swir
        self.trigger = trigger

    def validate (self):
        def _is_valid_gt ():
            #extract gt raster images from path
            gt_list = [os.path.join(self.gt_path, f) for f in os.listdir(self.gt_path) if f.endswith(self.rst_ext)]
            #check if all images are single band
            for gt_rst in gt_list:
                _gt_tst = rxr.open_rasterio(gt_rst, chunks= (1,500, 500))
                assert _gt_tst.rio.count == 1, 'Only single-band rasters are allowed for ground truth'
            return gt_list
        
        def _is_valid_target ():
            #extract target raster images from path
            target_list = [os.path.join(self.target_path, f) for f in os.listdir(self.target_path) if f.endswith(self.rst_ext)]
            #if swir path is provided, check if band count is equal to provided bands in target_bands and target_bands_swir
            if self.target_path_swir != None:
                target_swir_list = [os.path.join(self.target_path_swir, f) for f in os.listdir(self.target_path_swir) if f.endswith(self.rst_ext)]
                for vnir_rst in target_list:
                    _vnir_rst = rxr.open_rasterio(vnir_rst, chunks= (1,500, 500))
                    assert _vnir_rst.rio.count == len(self.target_bands), f'Number of bands in {vnir_rst} different from informed labels in target_bands ({str(self.target_bands)})'
                for swir_rst in target_swir_list:
                    _swir_rst = rxr.open_rasterio(swir_rst, chunks= (1,500, 500))
                    assert _swir_rst.rio.count == len(self.target_bands_swir), f'Number of bands in {swir_rst} different from informed labels in target_bands_swir ({str(self.target_bands_swir)})'
                target_list = list(zip(target_list, target_swir_list))
            #if swir is stacked, check if band count is equal to provided bands in target_bands
            else:
                for bands_rst in target_list:
                    _bands_rst = rxr.open_rasterio(bands_rst, chunks= (1,500, 500))
                    assert _bands_rst.rio.count == len(self.target_bands), f'Number of bands in {bands_rst} different from informed labels in target_bands ({str(self.target_bands)})'
            return target_list
        
        assert self.ini_file.endswith('.ini') == True, "Use Water Detect default .ini file"
        assert type(self.max_cluster) == tuple and len(self.max_cluster) == 3, "Pass max_cluster as a tuple (min,max,step). Ex: max_cluster = (3,9,1)"
        assert type(self.reg) == tuple and len(self.reg) == 3, "Pass reg as a tuple (min,max,step). Ex: reg = (0.01,0.1,0.01)"
        assert self.acc_metric == 'mcc' or self.acc_metric == 'f1' or self.acc_metric == 'kappa', 'Perfomance metrics can only be mcc, f1 or kappa'
        #return list of gt and target rasters if all parameters are valid
        gt_list = _is_valid_gt ()
        target_list = _is_valid_target ()
        return gt_list, target_list
    
    def combinations (self):
        gt_list, target_list = self.validate()
        #zip lists to match each gt with target raster
        if self.target_path_swir != None:
            tg_lst, sw_lst = zip(*target_list)
            input_raster_list = list(zip(gt_list, tg_lst, sw_lst))
        else:
            input_raster_list = list(zip(gt_list, target_list))
        #calculated all combinations between max_cluster and regularization
        max_cluster_list = list(range(self.max_cluster[0], self.max_cluster[1]+self.max_cluster[2], self.max_cluster[2]))
        regularization_list = [round(x, 2) for x in list(np.arange(self.reg[0], self.reg[1]+self.reg[2], self.reg[2]))]     
        ini_combination = [(x,y) for x in max_cluster_list for y in regularization_list]
        return input_raster_list, ini_combination
    
    def change_ini (self, max_cluster_vlue, reg_vlue):
        #Change ini values
        a_file = open(self.ini_file, "r")
        list_of_lines = a_file.readlines()
        list_of_lines[117] = 'regularization = ' + str(reg_vlue) + '\n'
        list_of_lines[124] = 'max_clusters = ' + str(max_cluster_vlue) + '\n'
        #check if vnir (BGR-NIR) or vnir-swir(BGR-NIR-SWIR1-SWIR2)
        if len(self.target_bands) == 4:
            list_of_lines[84] = "#\t\t    ['mndwi', 'ndwi', 'Mir2'],\n"
            list_of_lines[104] = "\t\t    ['ndwi', 'Nir' ],\n"
        elif len(self.target_bands) == 6 or self.target_path_swir != None:
            list_of_lines[84] = "\t\t    ['mndwi', 'ndwi', 'Mir2'],\n"
            list_of_lines[104] = "#\t\t    ['ndwi', 'Nir' ],\n"
        #open and edit default .ini file
        a_file = open(self.ini_file, "w")
        a_file.writelines(list_of_lines)
        a_file.close()
        return self.ini_file   
    
    def wd_mask (self, rst_target_path, rst_swir_path = None):
        def open_cube (self, rst_target_path, rst_swir_path = None):
            #read rasters and export bands as dicts
            div_factor=10000
            if self.target_path_swir != None:
                ds = rasterio.open(rst_target_path)
                cube = ds.read()
                arrays = {}
                out_shape = None
                for i, layer in enumerate(self.target_bands):
                    out_shape = cube[i].shape if out_shape is None else out_shape
                    arrays[layer] = ds.read(i+1, out_shape=out_shape).squeeze()/div_factor
                ds2 = rasterio.open(rst_swir_path)
                cube = ds2.read()
                for i, layer in enumerate(self.target_bands_swir):
                    out_shape = cube[i].shape if out_shape is None else out_shape
                    arrays[layer] = ds2.read(i+1, out_shape=out_shape).squeeze()/div_factor
                return arrays, ds
            else:
                ds = rasterio.open(rst_target_path)
                cube = xr.open_rasterio(rst_target_path)/div_factor
                arrays = {}             
                for i, layer in enumerate(self.target_bands):
                    arrays[layer] = cube[i].values
                return arrays, ds

        def to_xr (self, ds, wmask):
            #retrieve values from original raster
            res = ds.transform[0]
            xmin = ds.transform[2]
            ymax = ds.transform[5]
            ysize = wmask.water_mask.shape[0]
            xsize = wmask.water_mask.shape[1]
            #create an array that stores all possible coordinates
            y_np = np.arange(ymax, ymax - ysize * res, -res)
            x_np = np.arange(xmin, xmin + xsize * res, res)
            #data into dataset
            wd_xr = xr.DataArray(
                data= wmask.water_mask,
                coords = {'y': y_np, 'x': x_np},
                dims= ('y', 'x'),
            )
            return wd_xr

        def vnir_wd_mask (self, rst_target_path):
            #configure wd
            config = wd.DWConfig(config_file= self.ini_file)
            config.detect_water_cluster
            #open raster as arrays
            bands, ds = open_cube(self, rst_target_path)
            # As we don't have Mir band, it will cheat the water detect (to be corrected in the next version)
            bands['mndwi'] = np.zeros_like(bands['Green'])
            bands['mbwi'] = np.zeros_like(bands['Green'])
            bands['Mir2'] = np.zeros_like(bands['Green'])
            invalid_mask = (bands['Nir'] == 0)
            #run wd
            wmask = wd.DWImageClustering(bands=bands, bands_keys=['ndwi', 'Nir'], invalid_mask=invalid_mask, config=config, glint_processor=None)
            mask = wmask.run_detect_water()
            return to_xr (self, ds, wmask)

        def swir_wd_mask (self, rst_target_path, rst_swir_path=None):
            #configure wd
            config = wd.DWConfig(config_file= self.ini_file)
            config.detect_water_cluster
            #open raster as arrays
            bands, ds = open_cube(self, rst_target_path, rst_swir_path=rst_swir_path)
            invalid_mask = (bands['Mir2'] == 0)
            #run wd
            wmask = wd.DWImageClustering(bands=bands, bands_keys=['mndwi', 'ndwi', 'Mir2'], invalid_mask=invalid_mask, config=config, glint_processor=None)
            mask = wmask.run_detect_water()
            return to_xr (self, ds, wmask)
        
        #check if swir was passed as unique bands
        if self.target_path_swir != None:
            return swir_wd_mask (self, rst_target_path, rst_swir_path=rst_swir_path)
        #check if image has 4 bands to run vnir mask
        elif len(self.target_bands) == 4:
            return vnir_wd_mask (self, rst_target_path)
        #else run stacked swir
        else:
            return swir_wd_mask (self, rst_target_path)
        
    def retrieve_gt_points(self, rst_gt_path):

        #open raster as xarray
        gt_xr = xr.open_rasterio(rst_gt_path)
        #convert int boolean where only water/not-water are True
        ds_nan = xr.where((gt_xr==0) | (gt_xr==1), True, False)
        #convert coords to np flat
        x, y = np.meshgrid(gt_xr.coords['x'], gt_xr.coords['y'])
        #mask array with boolean
        x_masked  = np.ma.array(x, mask=ds_nan.values)
        y_masked  = np.ma.array(y, mask=ds_nan.values)
        #find target points and convert to xarray
        tgt_x= xr.DataArray(np.array(x_masked[x_masked.mask].data), dims="points")
        tgt_y= xr.DataArray(np.array(y_masked[y_masked.mask].data), dims="points")
        #retrieved gt coordinates for acc testing
        gt_xr = xr.where((gt_xr == 1), 1, 0)
        gt_xr = gt_xr.sel(x = tgt_x, y = tgt_y, method="nearest").squeeze('band')
        return gt_xr, tgt_x, tgt_y
    
    def mask_accuracy (self, gt_xr, wd_xr, tgt_x, tgt_y):
        #extract wd values at gt locations
        wd_xr = xr.where((wd_xr  == 1), 1, 0)
        wd_xr = wd_xr.sel(x = tgt_x, y = tgt_y, method="nearest")
        #check chosen acc metric
        if self.acc_metric == 'mcc':
            return matthews_corrcoef(gt_xr, wd_xr)
        elif self.acc_metric == 'f1':
            return f1_score(gt_xr, wd_xr)
        elif self.acc_metric == 'kappa':
            return cohen_kappa_score(gt_xr, wd_xr)
        else:
            raise ValueError('performance metric not found. Try mcc, f1 or kappa')
    
    def export_acc_to_csv (self, status, acc_result, max_cluster_vlue, reg_vlue, rst_gt_path, rst_target_path, out_folder):
        #retrieve rasters name
        rst_gt_path = os.path.basename(rst_gt_path)
        rst_target_path = os.path.basename(rst_target_path)
        #create dataframe with results
        df = pd.DataFrame({'acc_value': [acc_result], 'max_cluster': [max_cluster_vlue],
                   'regularization': [reg_vlue], 'acc_metric': ['mcc'], 'gt_rst_name': [rst_gt_path],
                   'target_rst_name': [rst_target_path] })
        #check status
        if status == 'final':
            df = pd.read_csv(os.path.join(out_folder, 'partial_sensitivity_results.csv'))
            df.sort_values(by='acc_value', ascending=False, inplace=True)
            df = df.head(1)            
        #export df to csv
        if os.path.isfile(os.path.join(out_folder, status + '_sensitivity_results.csv')):
            df.to_csv(os.path.join(out_folder, status + '_sensitivity_results.csv'), mode='a', index = False, header=False)
        else:
            df.to_csv(os.path.join(out_folder, status + '_sensitivity_results.csv'), mode='w', index = False, header=True)
        return df
    
    def export_wd_mask_raster (self, df, rst_target_path, out_folder, rst_swir_path = None):
        #create output file
        out_dir = os.path.join(out_folder, 'wd_sensitivy_results')                   
        def makemydir(out_dir):
            ##Create output file
            try:
                os.mkdir(out_dir)
            except OSError as error:
                print(error)
        makemydir (out_dir)
        #get final parameters from csv
        max_cluster_vlue = df['max_cluster'].head(1).values[0]
        reg_vlue = df['regularization'].head(1).values[0]
        #change .ini file
        self.ini_file = self.change_ini (max_cluster_vlue, reg_vlue)
        #export raster
        if rst_swir_path != None:
            wd_xr = self.wd_mask (rst_target_path, rst_swir_path = rst_swir_path)
            wd_xr.rio.to_raster(os.path.join(out_dir, 'mc_' + str(max_cluster_vlue) + '_reg_' + 
                                    str(reg_vlue) + '_' + os.path.basename(rst_target_path)[:-4] + '_swir' + self.rst_ext) )
        else:
            wd_xr = self.wd_mask (rst_target_path)
            wd_xr.rio.to_raster(os.path.join(out_dir, 'mc_' + str(max_cluster_vlue) + '_reg_' + 
                                    str(reg_vlue) + '_' + os.path.basename(rst_target_path)[:-4] + self.rst_ext) )
    
    def estimate_time (self, total_iterations, t_begin, t_end, est_time_lst):
        #calculate time to execute one iteration
        est_time_lst.append(t_end-t_begin)
        est_time_lst_av = sum(est_time_lst) / len(est_time_lst)  
        total_est_seconds = total_iterations * est_time_lst_av
        #convert total seconds to h,m,s
        m, s = divmod(total_est_seconds, 60)
        h, m = divmod(m, 60)
        if h != 0:
            if h == 1:
                print(f"Estimated remaining time: {h:0.0f} hour, {m:0.0f} minutes and {s:0.0f} seconds")
            else:
                print(f"Estimated remaining time: {h:0.0f} hours, {m:0.0f} minutes and {s:0.0f} seconds")
        else:
            print(f"Estimated remaining time: {m:0.0f} minutes and {s:0.0f} seconds")
        #update total_iterations
        total_iterations -= 1
        return total_iterations    
    
    def sensitivity (self):
        input_raster_list, ini_combination = self.combinations()
        out_folder = os.path.dirname(os.path.abspath(self.ini_file))
        total_iterations = len(input_raster_list) * len(ini_combination)
        est_time_lst = []
        
        if self.target_path_swir != None:
            for rst_gt_path, rst_target_path, rst_swir_path in input_raster_list:
                gt_xr, tgt_x, tgt_y = self.retrieve_gt_points(rst_gt_path)

                for max_cluster_vlue, reg_vlue in ini_combination:
                    print('Running...', os.path.basename(rst_target_path), 'with max cluster =', str(max_cluster_vlue),
                         'and reg =', str(reg_vlue))
                    
                    t_begin = time.perf_counter()
                    self.ini_file = self.change_ini (max_cluster_vlue, reg_vlue)
                    wd_xr = self.wd_mask (rst_target_path, rst_swir_path = rst_swir_path)
                    acc_result = self.mask_accuracy (gt_xr, wd_xr, tgt_x, tgt_y)
                    status = 'partial'
                    self.export_acc_to_csv (status, acc_result, max_cluster_vlue, reg_vlue, rst_gt_path, rst_target_path, out_folder)
                    clear_output()
                    t_end = time.perf_counter()
                    total_iterations = self.estimate_time (total_iterations, t_begin, t_end, est_time_lst)
                    
                    if acc_result >= self.trigger:
                        break
                status = 'final'
                df = self.export_acc_to_csv (status, acc_result, max_cluster_vlue, reg_vlue, rst_gt_path, rst_target_path, out_folder)
                self.export_wd_mask_raster (df, rst_target_path, out_folder, rst_swir_path)
                clear_output()                    
            clear_output()
            print('Done!')            
        else:
            for rst_gt_path, rst_target_path in input_raster_list:
                gt_xr, tgt_x, tgt_y = self.retrieve_gt_points(rst_gt_path)
                for max_cluster_vlue, reg_vlue in ini_combination:
                    print('Running...', os.path.basename(rst_target_path), 'with max cluster =', str(max_cluster_vlue),
                         'and reg =', str(reg_vlue))
                    
                    t_begin = time.perf_counter()
                    self.ini_file = self.change_ini (max_cluster_vlue, reg_vlue)
                    wd_xr = self.wd_mask (rst_target_path)
                    acc_result = self.mask_accuracy (gt_xr, wd_xr, tgt_x, tgt_y)
                    status = 'partial'
                    self.export_acc_to_csv (status, acc_result, max_cluster_vlue, reg_vlue, rst_gt_path, rst_target_path, out_folder)
                    clear_output()
                    t_end = time.perf_counter()
                    total_iterations = self.estimate_time (total_iterations, t_begin, t_end, est_time_lst)
                    
                    if acc_result >= self.trigger:
                        break
                status = 'final'
                print('Exporting most accurate raster...')
                df = self.export_acc_to_csv (status, acc_result, max_cluster_vlue, reg_vlue, rst_gt_path, rst_target_path, out_folder)
                self.export_wd_mask_raster (df, rst_target_path, out_folder)
                clear_output()
            clear_output()
            print('Done!')