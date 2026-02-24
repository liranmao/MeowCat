# ================== RCTD deconvolution (Seurat v5 reference: MainType) ==================
library(spacexr)
library(Matrix)
library(Seurat)
# library(stringr)  # not used; safe to drop

set.seed(42)

# ---------- helpers ----------
collapse_duplicate_genes <- function(mat) {
  if (anyDuplicated(rownames(mat))) {
    mat <- as(rowsum(as.matrix(mat), group = rownames(mat)), "dgCMatrix")
  }
  mat
}

read_positions_visium <- function(spatial_dir) {
  tp_csv  <- file.path(spatial_dir, "tissue_positions.csv")
  tp_list <- file.path(spatial_dir, "tissue_positions_list.csv")
  f <- if (file.exists(tp_csv)) tp_csv else tp_list
  stopifnot(file.exists(f))
  df <- tryCatch(read.csv(f, header = TRUE),
                 error = function(e) read.csv(f, header = FALSE))
  if (is.null(colnames(df)) || all(colnames(df) == paste0("V", seq_len(ncol(df))))) {
    colnames(df) <- c("barcode","in_tissue","array_row","array_col",
                      "pxl_row_in_fullres","pxl_col_in_fullres")
  }
  df
}

# Build SpatialRNA from a Visium sample directory that has 10x outputs
read_visium_as_spatialRNA <- function(sample_dir) {
  mtx_dir   <- file.path(sample_dir, "filtered_feature_bc_matrix")
  mtx_path  <- file.path(mtx_dir, "matrix.mtx.gz")
  feats_path<- file.path(mtx_dir, "features.tsv.gz")
  barc_path <- file.path(mtx_dir, "barcodes.tsv.gz")
  stopifnot(file.exists(mtx_path), file.exists(feats_path), file.exists(barc_path))

  M        <- Matrix::readMM(mtx_path)  # RAW ST UMI COUNTS (10x filtered_feature_bc_matrix)
  feat_df  <- read.delim(feats_path, header = FALSE, stringsAsFactors = FALSE)
  barcodes <- read.delim(barc_path,  header = FALSE, stringsAsFactors = FALSE)[[1]]

  # Keep only Gene Expression features; use gene symbols (your V2)
  if (ncol(feat_df) >= 3) {
    keep <- feat_df[[3]] == "Gene Expression"
    M       <- M[keep, , drop = FALSE]
    feat_df <- feat_df[keep, , drop = FALSE]
  }
  # Your feat_df shows V2 = gene symbols; prefer that, else fallback to V1
  gene_symbols <- if (ncol(feat_df) >= 2 && any(nzchar(feat_df[[2]]))) feat_df[[2]] else feat_df[[1]]
  rownames(M) <- gene_symbols
  colnames(M) <- barcodes

  counts <- as(M, "dgCMatrix")
  counts <- collapse_duplicate_genes(counts)     # sum duplicated symbols if present

  # coordinates, keep in-tissue
  pos <- read_positions_visium(file.path(sample_dir, "spatial"))
  pos <- subset(pos, in_tissue == 1)
  coords <- pos[, c("pxl_col_in_fullres","pxl_row_in_fullres")]
  rownames(coords) <- pos$barcode
  colnames(coords) <- c("x","y")

  # align barcodes
  bc <- intersect(colnames(counts), rownames(coords))
  counts <- counts[, bc, drop = FALSE]
  coords <- coords[bc, , drop = FALSE]

  SpatialRNA(coords, counts)
}

# simple plotting wrapper with robust cutoff fallback
plot_weights <- function(cell_type_names, puck, outfile_pdf, weights) {
  pdf(outfile_pdf)
  for (cell_type in cell_type_names) {
    cutoff <- tryCatch(spacexr:::UMI_cutoff(puck@nUMI),
                       error = function(e) 0)  # fallback if internal fn changes
    my_cond <- weights[, cell_type] > cutoff
    plot_var <- weights[, cell_type]; names(plot_var) <- rownames(weights)
    if (sum(my_cond) > 0) {
      p <- plot_puck_wrapper(
        puck, plot_var, NULL,
        minUMI = 100, maxUMI = 200000,
        min_val = 0, max_val = 1,
        title = cell_type, my_cond = my_cond
      )
      print(p)
    }
  }
  dev.off()
}

# ---------- paths ----------
base_dir <- "/project/KidneyHE/data_lung/other_states/"
sample_dirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)
sample_dirs <- sample_dirs[grepl("/P[^/]+$", sample_dirs)]  # folders starting with "P"

# ---------- load single-cell and prepare references per group ----------
singlecell_all <- readRDS("/project/CATCH/liran/he_anno/data_lung/single_cell/snRNA_all_in_one.to_Mingyao_Group.rds")

stopifnot("MainType" %in% colnames(singlecell_all@meta.data))
stopifnot("Group_ID" %in% colnames(singlecell_all@meta.data))

DefaultAssay(singlecell_all) <- "RNA"

# Groups you care about
groups_of_interest <- c("LUAD", "MIA", "AIS", "AAH", "Normal")

# Helper to build a reference for one group
build_reference_for_group <- function(singlecell_all, group_name, min_UMI = 10) {
  message("Building reference for group: ", group_name)

  sc_sub <- subset(singlecell_all, Group_ID == group_name)
  if (ncol(sc_sub) == 0) {
    stop("No cells found in singlecell object for group: ", group_name)
  }

  DefaultAssay(sc_sub) <- "RNA"

  # Seurat v5: pull counts from the RNA assay "counts" layer
  counts_sc <- LayerData(sc_sub$RNA, layer = "counts")  # RAW scRNA UMI COUNTS (features x cells)
  counts_sc <- as(counts_sc, "dgCMatrix")

  # Labels
  labels <- sc_sub@meta.data[colnames(counts_sc), "MainType"]
  names(labels) <- colnames(counts_sc)

  # Ensure gene symbols & collapse duplicates if needed
  counts_sc <- collapse_duplicate_genes(counts_sc)

  # Build RCTD reference
  reference_major <- Reference(counts_sc, labels, min_UMI = min_UMI)
  message("Reference for ", group_name, " built with ",
          ncol(counts_sc), " cells and ",
          nrow(counts_sc), " genes across ",
          length(levels(labels)), " cell types.")

  return(reference_major)
}

# Lazy cache of references so we build each group only once
reference_cache <- list()

# Helper: infer group from sample_id like "P1_LUAD", "P10_MIA", "P4_Normal", "P23_AIS1"
infer_group_from_sample_id <- function(sample_id, groups) {
  hits <- groups[sapply(groups, function(g) grepl(g, sample_id, fixed = TRUE))]
  if (length(hits) == 1) return(hits)
  if (length(hits) == 0) {
    warning("Could not infer group from sample_id: ", sample_id)
  } else {
    warning("Multiple groups matched for sample_id: ", sample_id,
            " -> ", paste(hits, collapse = ", "))
  }
  return(NA_character_)
}

# ---------- run per-sample ----------
for (sample_dir in sample_dirs) {
  sample_id <- basename(sample_dir)
  message("\n===== RCTD on sample: ", sample_id, " =====")

  # Infer group from folder name (LUAD, MIA, AIS, AAH, Normal)
  sample_group <- infer_group_from_sample_id(sample_id, groups_of_interest)
  if (is.na(sample_group)) {
    message("Skipping sample ", sample_id, " because group could not be inferred.")
    next
  }
  message("Sample ", sample_id, " assigned to group: ", sample_group)

  # Get or build reference for this group
  if (!sample_group %in% names(reference_cache)) {
    reference_cache[[sample_group]] <- build_reference_for_group(singlecell_all, sample_group)
  }
  reference_major <- reference_cache[[sample_group]]

  # Output paths
  out_dir <- file.path(sample_dir, "deconvolution_rctd")
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  csv_out <- file.path(out_dir, "major_prop.csv")
  pdf_out <- file.path(out_dir, "major_prop.pdf")

  if (file.exists(csv_out)) {
    message("Already done: ", csv_out, " — skipping.")
    next
  }

  # Build SpatialRNA from 10x files
  puck <- read_visium_as_spatialRNA(sample_dir)

  # Fit
  myRCTD <- create.RCTD(puck, reference_major, max_cores = 5)
  myRCTD <- run.RCTD(myRCTD, doublet_mode = "full")

  # Normalize to proportions, save CSV + PDF
  nw <- normalize_weights(myRCTD@results$weights)
  rownames(nw) <- colnames(myRCTD@spatialRNA@counts)
  nw <- as.matrix(nw)

  write.csv(nw, csv_out, quote = FALSE)
  cell_type_names <- myRCTD@cell_type_info$info[[2]]
  plot_weights(cell_type_names, myRCTD@spatialRNA, pdf_out, nw)

  message("Saved: ", csv_out, " and ", pdf_out)

  rm(puck, myRCTD, nw, cell_type_names)
  gc()
}
