################### to be implemented ###################################

rm(list=ls())
suppressPackageStartupMessages({
  library(ggplot2)
  library(getopt)
  library(harmony)
})
start.time <- Sys.time()
spec <- matrix(
  c("read_path", "f",1,"character","file folder of data", 
    'save_dir',"s",1,"character","where to save result",

  ),
  byrow=TRUE, ncol=5)
opt <- getopt(spec=spec,debug=FALSE)
#print(opt)
# PCA implementation,

# saveRDS(pca_mat,file="./pca_mat.rds")
# saveRDS(meta_data,file="./meta_data.rds")
# I can calculate PCA ahead of time
pca_mat = readRDS()
meta_data = readRDS()
harmony_mat = RunHarmony(pca_mat,meta_data, "BATCH",plot_convergence = F,max.iter.harmony=50)
saveRDS(harmony_mat)































# rm(list=ls())
# method="harmony"
# suppressPackageStartupMessages({
#   library(Seurat)
#   library(harmony)
#   library(SingleCellExperiment)
#   library(ggplot2)
# })

# ####################### parameter setting ###############################
# #setwd("/Users/yxkang/Desktop/Medium/Harmony/")
# data <- readRDS("/project/MultiSampleIstar/dataset/bct/bct_raw.rds")
# ####################### parameter setting ###############################

# print(table(colData(data)$BATCH , colData(data)$celltype))
# print("=====================================")
# print(data)
# data_seurat=CreateSeuratObject(counts = counts(data),meta.data = as.data.frame(colData(data)))
# data_seurat <- NormalizeData(data_seurat, verbose = FALSE)
# data_seurat <- FindVariableFeatures(data_seurat, selection.method = "vst", nfeatures = 2000, verbose = FALSE)

# # Run the standard workflow for visualization and clustering
# data_seurat <- ScaleData(data_seurat, verbose = FALSE)
# data_seurat <- RunPCA(data_seurat, npcs = 30, verbose = F)
# #data_seurat <- RunUMAP(data_seurat, reduction = "pca", dims = 1:30, verbose = F)
# #DimPlot(data_seurat,reduction = "umap",group.by = "BATCH") + plot_annotation(title = "data before integration")


# ############################################################################################
# ############################################################################################
# ############################################################################################
# ############################################################################################
# start_time <- Sys.time()

# pca_mat = data_seurat@reductions$pca@cell.embeddings
# meta_data = data_seurat@meta.data

# saveRDS(pca_mat,file="./pca_mat.rds")
# saveRDS(meta_data,file="./meta_data.rds")

# harmony_mat = RunHarmony(pca_mat,meta_data, "BATCH",plot_convergence = F,max.iter.harmony=50)
# data_seurat[["harmony"]] <- CreateDimReducObject(
#   embeddings = harmony_mat,        # N_cells × N_dim  matrix
#   key = "harmony_",                # prefix for each dimension
#   assay = DefaultAssay(data_seurat) # \
# )

# end_time <- Sys.time()
# # Calculate execution time
# execution_time <- end_time - start_time
# sprintf("runing harmony cost: %ds", round(execution_time))
# ############################################################################################
# ############################################################################################
# ############################################################################################
# ############################################################################################

# #data_seurat <- RunTSNE(data_seurat, reduction = "harmony", dims = 1:30, verbose = F)
# data_seurat <- RunUMAP(data_seurat, reduction = "harmony", dims = 1:30, verbose = F)
# # data_seurat <- FindNeighbors(data_seurat, reduction = "harmony", dims = 1:30,verbose=FALSE)
# # data_seurat <- FindClusters(data_seurat,verbose=FALSE,resolution = 0.4)

# # p1=DimPlot(data_seurat, reduction = "tsne", group.by = "BATCH", label.size = 10)+ggtitle("Integrated Batch")
# # p2=DimPlot(data_seurat, reduction = "tsne", group.by = "celltype",label.size = 10)+ggtitle("Integrated Celltype")
# # p= p1 + p2 
# # print(p)
# # print("harmony tsne done")
# # ggsave("harmony_pancreas_tsne.png",p)

# #saveRDS(data_seurat,file="harmony.rds")

# p3=DimPlot(data_seurat, reduction = "umap", group.by = "BATCH", label.size = 10)+ggtitle("Integrated Batch")
# p4=DimPlot(data_seurat, reduction = "umap", group.by = "celltype",label.size = 10)+ggtitle("Integrated Celltype")
# p= p3 + p4
# print(p)
# print("harmony umap done")
# ggsave("harmony_bct_umap.png",p)
# #saveRDS(data_seurat,file="harmony.rds")
