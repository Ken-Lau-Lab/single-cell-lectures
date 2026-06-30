import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
import re
import pickle
#this is the file in the home directory

#import sys
#sys.path.append('/home/lucy/')
#import import_ipynb
#import zc_functions_ln as zc





def normalization( dat_ct, save_arcsinh = True):
    """this function normalize the data so that each cell has the same 
    number of total counts as the median value of the total counts among all cells.
    The data will also be log-like transformed
    Count values will also be transformed to z-scores for each gene"""
    sc.pp.normalize_total(dat_ct) 
    dat_ct.X = np.arcsinh(dat_ct.X).copy()
    if( save_arcsinh):
        dat_ct.layers['arcsinh'] = dat_ct.X.copy()
    
    sc.pp.scale(dat_ct)
    
    return dat_ct

#from SEPPP1 project 2_visual_cytotrace notebook
def scale_column( df, col_name, scaled_col_name):
    """This function scaled a column in a dataframe so that all values is between 0 and 1 
    a new dataframe with the scaled column added will be returned """
    min_value = min( df[col_name])
    max_value = max(df[col_name])
    df[scaled_col_name] = (df[col_name] - min_value) / (max_value - min_value)
    
    return df

def differential_gene_result(adata, sample_id, num_genes, group_name = "score_classification" , plot = True, gene_group_key = 'metap_stem_diff'):
    
    """this function returns a dataframe of differential genes calculated between 
    groups of one sample in the dataset separated within the sample based on the l
    abel of a obs variable column specified by the group_name param.
    A resultant dataframe can have number of genes specified by the num_genes param"""
    
    sc.tl.rank_genes_groups(adata, group_name, method = 't-test', use_raw = False, key_added =gene_group_key )
    if(plot):
        sc.pl.rank_genes_groups(adata, n_genes=20, sharey = False, key = gene_group_key)
    gene_names = pd.DataFrame( adata.uns[gene_group_key]["names"] ) 
    gene_names = gene_names[0:num_genes]
    gene_names["sample_id"] = [sample_id for i in range(num_genes)]
    return gene_names
    

    
def differential_gene_result(adata, num_genes, group_name = "score_classification" , sample_id = None, plot = True, gene_group_key = 'metap_stem_diff'):
    
    """this function returns a dataframe of differential genes calculated between 
    groups of one sample in the dataset separated within the sample based on the l
    abel of a obs variable column specified by the group_name param.
    
    A resultant dataframe can have number of genes specified by the num_genes param
    
    sample_id: if not none, a column of sample_id will be added to the resultant dataframe. 
        This can be helpful if multiple such dataframe from different samples need to be merged for later analysis"""
    
    sc.tl.rank_genes_groups(adata, group_name, method = 't-test', use_raw = False, key_added =gene_group_key )
    if(plot):
        sc.pl.rank_genes_groups(adata, n_genes=20, sharey = False, key = gene_group_key)
    gene_names = pd.DataFrame( adata.uns[gene_group_key]["names"] ) 
    gene_names = gene_names[0:num_genes]
    if sample_id:
        gene_names["sample_id"] = [sample_id for i in range(num_genes)]
    return gene_names
    






def clustering(dat, n_pcs = 50, n_neighbors= None,  use_highly_variable = False, highly_variable_layer = None, flavor = 'seurat', neighbor_distance_metrics='euclidean', raw_layer_name = 'raw_counts', verbose = False, norm_raw_before_arcsinh = True):
    """This function calculate pcs, and get the umap of the dataset
    dat will be edited in place. NOTE: the dataset need to have a raw count layer if want to run highly variable gene
    
    param dat: the anndata object that has transformed data
    param n_pcs: number of PCs to pass in to sc.tl.pca() function
    param n_neighbors: number of neightbors to pass in to the sc.pp.neighbors() function
    param use_highly_variable: if run PCA with only highly variable genes
    param highly_variable_layer: the count matrix layer used to run highly variable gene, if None, arcsinh used 
    param flavor: flavor for highly variable gene function, see scanpy highly variable gene function for more detail, need to change layer accordingly
    param neighbor_distance_metrics: distance type when running the neighbor function, default euclidean distance (can use cosine which is seurat's default)
    param raw_layer_name the name of the layer that has the raw count
    param use_raw_before_arcsinh: whether normalize the raw count library size before arcsinh transformation or not, default set to True 
    
    return None
    """
    
    if(n_neighbors == None):
        n_neighbors = int( np.sqrt(dat.n_obs) )
        
    if(use_highly_variable):
        if ('highly_variable' in dat.var ):
            if(verbose): print("Running PCA with hvg")
            sc.tl.pca(dat, return_info=False , use_highly_variable = True)
        else: 
            # run highly variable gene from here 
            if( raw_layer_name not in dat.layers):
                #need a raw layer, else will give error
                print(" Error: need to have/specify a raw count layer")
                return 
            if(verbose): print("Transforming data")
            dat.layers["curr_layer" ] = dat.X.copy() # save the current layer
            dat.X = dat.layers[raw_layer_name].copy() # change to the raw layer 
            add_highly_variable_gene_col(dat, layer = highly_variable_layer, flavor = flavor, use_normed= norm_raw_before_arcsinh) # run highly variable gene function
            
            if(verbose): print("Running PCA with hvg")
            sc.tl.pca(dat, return_info=False , use_highly_variable = True)
            dat.X = dat.layers["curr_layer" ].copy() # change back to current layer 
            del dat.layers["curr_layer" ] #clean up
    else:
        if(verbose): print("Running PCA without hvg")
        sc.tl.pca(dat, return_info=False, use_highly_variable = False )
    
    if(verbose): print("Calculating Neighborhood")
    sc.pp.neighbors( dat, n_neighbors= n_neighbors , n_pcs=n_pcs, metric=neighbor_distance_metrics)
    if(verbose): print("Computing UMAP")
    sc.tl.umap(dat)
    
    return 




def pct_dropout(dat):
    """This function adds a 'pct_dropout_by_counts' column to the anndata's var layer. The value is in percentage ( so a value of 2 == 2% dropout """
    dat.var["pct_dropout_by_counts"] = np.array(
            (1 - (dat.X.astype(bool).sum(axis=0) / dat.n_obs)) * 100
        ).squeeze()
    
def add_highly_variable_gene_col(dat, layer = None, flavor = 'seurat', use_normed = True):
    """ This function find highly variable genes for a dataset and add the variable in-place
    @param dat: anndata object with raw count
    @param layer: the layer of X want to use, use arcsinh is not specified
    @param flavor: flavor of method, see scanpy's documentation of the highly variable gene function
    
    @return None"""
    
    if (not layer):
        if(use_normed):
            X_norm = sc.pp.normalize_total(dat, inplace=False)['X']
            dat.layers["arcsinh"] = np.arcsinh(X_norm )
        else:
            dat.layers["arcsinh"] = np.arcsinh(dat.X.copy() )
        
    sc.pp.highly_variable_genes(dat, layer="arcsinh", flavor=flavor, inplace = True)
    
    return 







#from notebook ~/plasticity/6_integrate_NL.ipynb
def parse_rank_gene_group_dict(rank_gene_dict, fields_to_keep = ['logfoldchanges', 'pvals_adj', 'pvals', 'names', 'scores'], 
                               condition = 'fetal', sort_by = 'pvals_adj', ascending = True):
    """this function parse the dictionary output by the scanpy function scanpy.tl.rank_gene_group (differential gene expression) into a pd dataframe.
    this function only parse one condition (eg. one cell type's DGE compare with another condition )
    
    @param rank_gene_dict: the dictionary output by the rank_gene_group function (stored in the anndata's uns field, default 'rank_gene_groups'
    @param field to keep: fields in the dictionary keys to keep as columns in the resultant dataframe
    @param condition: if the rank gene group function is performed over >2 groups (eg. DGE between different cell types), which condition's DGE do you want to extract from the dictionary )
    
    return: the dataframe parsed from the dictionary that contains the fields_to_keep terms as columns"""
    
    result_df = pd.DataFrame({fields_to_keep[i] : rank_gene_dict[fields_to_keep[i]][condition] for i in range(len(fields_to_keep))
              })
    result_df.sort_values(by = sort_by, inplace=True, ascending=ascending)
    
    return result_df





    
    

    
# this is a random note, but how to convert dict with different length of values to a pd dataframe?
# pd.DataFrame(dict([(col_name,pd.Series(values)) for col_name,values in my_dict.items() ]))

def dictToDf(the_dict):
    """ so this function converts a dictionary of different lengths of values to a dataframe with the dict key as columns names and the values in each column  """
    df = pd.DataFrame( dict([(col_name,pd.Series(values)) for col_name,values in the_dict.items() ]) )
    return df


# another way to save dictionary is to save to a pickle file

def save_dict_pickle( to_save_dict, path_name): 
    """this function saves a dictionary to a pickle file
    @param to_save_dict: this is the python dictionary object to be saved
    @param path_name: location and file names to save the pickle file. Should ends with .pkl ( eg. ~/Dropbox/hi.pkl)
    @return None"""
    # Save the dictionary to a file
    with open(path_name, 'wb') as f:
        pickle.dump(to_save_dict, f)
    return 

def load_pickle_to_dict( file_name,):
    """ load a dictionary object that is saved in a pkl file
    @param file_name: path and filename to find the file, should ends with .pkl
    @return: the dictionary object loaded from the file"""
    
    with open(file_name, 'rb') as file:
    # Load the dictionary from the file
        my_dict = pickle.load(file)

    return my_dict
    


    
# from ~/plasticity/9_fetal_atlas... notebook    
def get_up_down_dge_from_rankGenDict(adata, rank_gene_group_key, 
                                     group_by, sub_group = None, num_genes = 20, sort_by = ['pvals_adj'], ascending = [True]):
    """this function get top up and down regulated genes from each group in output by scanpy's rank_gene_group (differential gene expression) function, return a dataframe as described below
    @param adata: the anndata object
    @param rank_gene_group_key: the name of rank_gene_group output stores adata's uns field (the names input as argument to 'key_added' parameter in the rank_gene_group function )
    @param group_by: the obs column name input as the 'group_by' argument in the sc.tl.rank_gene_group function
    @param sub_group: a subset of adata.obs[group_by] list (if interested in only some group's comparison
    @param num_genes: number of top significant genes output from this function
    
    @return: a dataframe with <group_name>_up and <group_name>_down as column names and significant differential genes in each column"""
    
    
    rank_gene_dict = adata.uns[rank_gene_group_key]
    if(not sub_group):
        sub_group = adata.obs[group_by].unique()
    
    all_df = pd.DataFrame()
    for g in sub_group:
        dge_df = parse_rank_gene_group_dict(adata.uns[rank_gene_group_key], fields_to_keep = ['logfoldchanges', 'pvals_adj', 'pvals', 'names', 'scores'], 
                               condition = g, sort_by = sort_by, ascending = ascending)
        g_up = dge_df[dge_df["logfoldchanges"] > 0].sort_values(by = sort_by, ascending = ascending)
        g_down = dge_df[dge_df["logfoldchanges"] < 0].sort_values(by = sort_by, ascending = ascending)
        
        all_df[f"{g}_up"] = g_up["names"][0:num_genes].values
        all_df[f"{g}_down"] = g_down["names"][0:num_genes].values
        
    return all_df




def check_gene_present( dat, potnetial_gene_name, check_prefix_bol, prefix_digit, print_prefix_gene_bol, is_anndata = True):
    """ this function checks 1) if  the given gene name is in the var_names field of the anndata, 2) if there are genes with the same prefix
    @param dat: the anndata whose var_names will be checked if the given gene name presents in 
    @param potential_gene_name: the gene name that we want to check if present in the dat
    @param check_prefix_bol: boolean value to specificy if we want to check whether there are genes who has the same prefix with the target gene
    @param prefix_digit: the number of letters to be extracted from the target gene str to be used as the prefix
    @param print_prefix_gene_bol: boolean value to specify if the list of genes with the same prefix with the target genes should be printed
    @param is_anndata: boolean to indicate if the dat object is an anndata object, if False, dat is expected to be a vector of gene names of type pd dataframe or series
    
    @return a boolean value if the target gene present in the dat
    """

    
    if( is_anndata ):
        gene_ls = dat.var_names
    else:
        gene_ls = dat
    
    ret_bol = np.isin(  potnetial_gene_name, gene_ls )
    print( f'{potnetial_gene_name} in dat: {ret_bol }' )

    if( check_prefix_bol):
        prefix_str = potnetial_gene_name[0: prefix_digit]
        found_df = gene_ls[ gene_ls.str.startswith( prefix_str )]
        print( f' {len( found_df) } genes found with the prefix {prefix_str }')
        if( print_prefix_gene_bol):
            print( found_df )


            
        
        
    
        
        
    
## ====================================================================
## some filter functions
## ====================================================================
#markers:
human_epi={
    'CT':['OTOP2','MEIS1'],
    'ABS':['KRT20','GUCA2A','ALDOB'],
    'SSC':['MSLN','MUC5AC','AQP5','TACSTD2','FSCN1','TFF2','ANXA1','ANXA10','REG4','MUC17','S100P','GSDMB','GSDMD','IL18','RELB','MDK','RARA','RXRA','AHR','AGRN','PDX1'],
    'ASC':['CLDN2','CD44','AXIN2','RNF43','TGFBI','EPHB2','TEAD2','CDX2'],
    'STM':['LGR5','OLFM4','ASCL2'],
    'TAC':['PCNA','MKI67'],
    'GOB':['ATOH1','MUC2','TFF3'],
    'EE':['CHGA','NEUROD1'],
    'TUF':['POU2F3','SOX9'],
}

human_nonepi={
    'T':['CD8A','CD3D','CD4','TRBC2','CD96','CD247'],
    'MYE':['CSF1R','CSF3R','CD14','MRC1'],
    'MAS':['KIT','KRT1'],
    'CD19':['CD19','CD79A','CD74','HLA-DRA','CD37','CD22','MS4A1'],
    'IGH':['IGHA1','IGHA2','JCHAIN'],
    'END':['VWF','MCAM'],
    'FIB':['COL1A1','FN1'],
}

mouse_non_epi = {'T':['Cd3d','Cd4','Trbc2','Cd96','Cd247'],
    'MYE':['Csf1r','Csf3r','Cd14','Mrc1'],
    'MAS':['Kit','Krt1'],
    'B':['Cd19', 'Ms4a1'],
    'PLAS':['Igha1','Igha2','Jchain'],
    'END':['Vwf','Mcam'],
    'FIB':['Col1a1','Fn1']}

mouse_epi_markers = {
 'CT': ['Otop2', 'Meis1'],
 'ABS': ['Krt20', 'Guca2a', 'Aldob'],
 'SSC': ['Muc5ac', 'Aqp5', 'Tacstd2', 'Tff2', 'Anxa1', 'Anxa10', 'Ly6a', 'S100a10'],
 'ASC': ['Cldn2', 'Cd44', 'Axin2', 'Rnf43', 'Tgfbi', 'Ephb2', 'Tead2', 'Cdx2'],
 'STM': ['Lgr5', 'Olfm4', 'Ascl2'],
 'TAC': ['Pcna', 'Mki67'],
 'GOB': ['Atoh1', 'Muc2', 'Tff3'],
 'EE': ['Chga', 'Neurod1'],
 'TUF': ['Pou2f3', 'Sox9', 'Dclk1'],
 'SQU': ['Krt14','Krt5','Krt6a','Fabp5'],
}

mouse_classic_markers = ['Lgr5', 'Pcna', 'Ptprc', 'Krt20', 'Muc2', 'Pou2f3', 'Dclk1', 'Chga', 'Lyz1','Krt14'] 
# stem, TA, immune (Leukocyte common antigen) , ABS, GOB, TUF, TUF, EE, PN

human_classic = ["KRT20", "LGR5", "OLFM4", "MKI67", "ATOH1", "MUC2","CHGA", "POU2F3"]

mescen_stromal_dict = { 'Cd81_FIB': ['Cd81', 'C3', 'Has1'], 'Cd90_FIB': ['Cd90','Eln', 'Magp', 'Col15a1'], 
            'Fgfr2_FIB': ['Fgfr2','Igfbp4', 'Igfbp3', 'Ces1d'],'Pdgfra_FIB': ['Pdgfra','Procr', 'Bmp2','Bmp5'],
            'PERI': ['Apold1', 'Rgs5', 'Des', 'Rgs4' ,'Ndufa4l2','Kcnj8', 'Rrad'], 'SMC':['Actg2', 'Myl9' , 'Lmod1' , 'Acta2', 'Tagln', 'Myh11' ] } # queried from Paerregaard et al. 2023 Nature Communications 

qc_strings =  ['total_counts', 'n_genes_by_counts',"pct_counts_in_top_200_genes", 'pct_counts_Mitochondrial', "pct_counts_ambient","dropkick_score", "dropkick_label", 'leiden']

qc_strings_no_dropkick =['total_counts', 'n_genes_by_counts',"pct_counts_in_top_200_genes", 'pct_counts_Mitochondrial', "pct_counts_ambient", 'leiden']

#from ~/immune_exclusion/1_integrate_9142... ipynb 6/17/22

def markers_plots_dict(dat, marker_dict, modify_key = None, modify_value = None, set_vmax = None, n_cols = 3):
    """This function visualize markers from a dictionary 
    where the key is the cell type and the values are the marker genes for the subtype
    
    @param dat: the anndata object whose data will be visualized
    @param marker_dict: a dictionary object with key of cell types and values of marker genes for the cell type
    @param modify_key: a dictionary key obejct contaning the keys from marker_dict whose value you want to modify
    @param modify_value: a dictionary object that contains the key in modify_key and values as the modified markers associated with the key 
    @param set_max: set maximum expression value for all plots """
    
    sc.set_figure_params(figsize=[3,3])
    for k,v in marker_dict.items():
        print("Subtype: "+ k)

        if( modify_key !=None and k in modify_key):
            v= modify_value[k]

        # use markers that present in the dataset gene field
        v = pd.Series( v)
        v_found = v[np.isin( v, dat.var_names)]
        v_unfound = v[~np.isin( v, dat.var_names)]
        print( f" genes absent in the dataset: {v_unfound}")
        if(len(v_found)<1):
            continue
        
        try:
            if(set_vmax):
                sc.pl.umap(dat,color= v_found ,use_raw=False,cmap='viridis', ncols=n_cols, vmax = set_vmax)
            else:
                sc.pl.umap(dat,color= v_found ,use_raw=False,cmap='viridis', ncols=n_cols)
        except KeyError as err:
            print(f"error:{err}")
            continue
    
    return


def add_mito_amb_col(dat, mito_str, amb_ls, mito_col_name = 'Mitochondrial', amb_col_name = 'ambient', run_qc = False):
    """adding the boolean var columns of mitocondrial and ambient genes
    the new anndata obj will be returned, but the passed in obj will not be updated
    retrun updated dat"""
    dat2 = dat.copy()
    dat2.var[mito_col_name] = dat.var.index.str.startswith(mito_str)
    dat2.var[amb_col_name] = np.isin(dat.var.index, amb_ls)
    if(run_qc):
        sc.pp.calculate_qc_metrics(dat2, qc_vars = [mito_col_name, amb_col_name], inplace=True)
    
    return dat2



def annotate_clusters(dat,  from_label, new_label, from_col ='leiden', to_col='cell_type', astype_str = True):
    """This function annotate clusters based on a existing anndata obs column, (NOT inplace)
    @param dat: the anndata object whose cluster you want to annotate
    @param from_label: a LIST of labels of the obs column that you want to annotate the to_col column based on (eg. leiden cluster 1,2,3)
    @param to_label: a STRING of a label of the obs column that will be annotated (eg. "stem cell" in cell_type column)
    @param from_col: the obs column that you want to annotate the to_col column based on
    @param to_col: the obs column that will be annotated
    @param astype_str: a boolean flag to force the labels to be string type 
    
    @return a new dataset that is annotated """

    col_df = pd.concat([dat.obs[from_col], dat.obs[to_col]], axis = 1)
    if(astype_str):
        col_df = col_df.astype(str)
    col_df[to_col][np.isin(col_df[from_col], from_label) ] = new_label
    dat2 = dat.copy()
    dat2.obs[to_col] = col_df[to_col]
    
    return dat2