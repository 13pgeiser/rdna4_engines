.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.amdhsa_code_object_version 6
.text
.protected gemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151
.globl gemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151
.p2align 8
.type gemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151,@function
gemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b128 s[28:31], s[0:1], 0x50
	v_lshrrev_b32_e32 v13, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v13
	v_and_b32_e32 v5, 31, v0
	s_mov_b32 s40, s2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s36, s26, 7
	s_lshr_b32 s36, s36, 3
	s_add_i32 s45, s27, 7
	s_lshr_b32 s45, s45, 3
	s_add_i32 s37, s36, s45
	s_add_i32 s45, s28, 7
	s_lshr_b32 s45, s45, 3
	s_add_i32 s38, s37, s45
	s_add_i32 s45, s29, 7
	s_lshr_b32 s45, s45, 3
	s_add_i32 s39, s38, s45
	s_cmp_ge_u32 s40, s39
	s_cbranch_scc1 .Lbf16w16_end
	s_cmp_eq_u32 s31, 0
	s_cbranch_scc1 .Lbf16w16_end
	s_cmp_gt_u32 s31, 16
	s_cbranch_scc1 .Lbf16w16_end
	s_cmp_lt_u32 s40, s38
	s_cselect_b64 s[4:5], s[12:13], s[14:15]
	s_cselect_b64 s[32:33], s[22:23], s[24:25]
	s_cselect_b32 s34, s28, s29
	s_cselect_b32 s35, s37, s38
	s_cmp_lt_u32 s40, s37
	s_cselect_b64 s[4:5], s[10:11], s[4:5]
	s_cselect_b64 s[32:33], s[20:21], s[32:33]
	s_cselect_b32 s34, s27, s34
	s_cselect_b32 s35, s36, s35
	s_cmp_lt_u32 s40, s36
	s_cselect_b64 s[4:5], s[8:9], s[4:5]
	s_cselect_b64 s[32:33], s[18:19], s[32:33]
	s_cselect_b32 s34, s26, s34
	s_cselect_b32 s35, 0, s35
	s_sub_i32 s42, s40, s35
	s_lshl_b32 s42, s42, 3
	s_add_i32 s42, s42, s41
	s_cmp_lt_i32 s42, s34
	s_cselect_b32 s56, 1, 0
	s_and_b32 s58, s30, 7
	s_add_i32 s43, s30, 0xff
	s_lshr_b32 s43, s43, 8
	s_add_i32 s44, s30, 31
	s_lshr_b32 s44, s44, 5
	s_lshl_b32 s59, s30, 1
	s_mul_i32 s46, s42, s59
	s_mul_hi_u32 s47, s42, s59
	s_add_u32 s4, s4, s46
	s_addc_u32 s5, s5, s47
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_lshl_b32 s53, s30, 2
	s_lshl_b32 s54, s34, 2
	s_lshl_b32 s57, s42, 2
	s_add_i32 s45, s41, 0
	s_mul_i32 s60, s45, s53
	s_add_i32 s45, s41, 8
	s_mul_i32 s61, s45, s53
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v12, 2, v5
	s_lshl_b32 s45, s41, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v9, s45, v12
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	s_mov_b32 s52, 0
	s_cmp_lg_u32 s58, 0
	s_cbranch_scc1 .Lbf16w16_scalar
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lbf16w16_fold
	.Lbf16w16_group:
	s_lshl_b32 s50, s52, 9
	s_lshl_b32 s51, s52, 10
	s_lshl_b32 s49, s52, 8
	s_sub_i32 s49, s30, s49
	s_lshr_b32 s48, s49, 3
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 0x100
	s_cselect_b32 s48, -1, s48
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s51, v3
	v_lshlrev_b32_e32 v1, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_s0t9_wdone
	s_mov_b32 exec_lo, s48
	buffer_load_b128 v[16:19], v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lbf16w16_s0t9_wdone:
	s_add_i32 s45, s41, 0
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w16_s0t0_nofill
	v_add_nc_u32_e32 v13, s60, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w16_s0t0_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_s0t0_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lbf16w16_s0t0_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lbf16w16_s0t0_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v48, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v49, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v50, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v51, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lbf16w16_s0t0_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v52, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v53, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v54, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v55, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lbf16w16_s0t0_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v56, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v57, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v58, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v59, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lbf16w16_s0t0_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v60, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v61, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v62, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v63, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lbf16w16_s0t0_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v64, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v65, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v66, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v67, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v31, v23
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lbf16w16_s0t0_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v68, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v69, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v70, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v71, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v31, v23
	.Lbf16w16_s0t0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w16_s0t0_tile_done:
	s_cmp_le_u32 s31, 8
	s_cbranch_scc1 .Lbf16w16_s0t1_tile_done
	s_add_i32 s45, s41, 8
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w16_s0t1_nofill
	v_add_nc_u32_e32 v13, s61, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w16_s0t1_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_s0t1_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v72, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v73, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v74, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v75, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v72, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v73, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v74, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v75, v31, v23
	s_cmp_le_u32 s31, 9
	s_cbranch_scc1 .Lbf16w16_s0t1_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v76, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v77, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v78, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v79, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v76, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v77, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v78, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v79, v31, v23
	s_cmp_le_u32 s31, 10
	s_cbranch_scc1 .Lbf16w16_s0t1_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v80, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v81, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v82, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v83, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v80, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v81, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v82, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v83, v31, v23
	s_cmp_le_u32 s31, 11
	s_cbranch_scc1 .Lbf16w16_s0t1_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v84, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v85, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v86, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v87, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v84, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v85, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v86, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v87, v31, v23
	s_cmp_le_u32 s31, 12
	s_cbranch_scc1 .Lbf16w16_s0t1_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v88, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v89, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v90, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v91, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s31, 13
	s_cbranch_scc1 .Lbf16w16_s0t1_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v92, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v93, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v94, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v95, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v31, v23
	s_cmp_le_u32 s31, 14
	s_cbranch_scc1 .Lbf16w16_s0t1_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v96, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v97, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v98, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v99, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v31, v23
	s_cmp_le_u32 s31, 15
	s_cbranch_scc1 .Lbf16w16_s0t1_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v100, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v101, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v102, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v103, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v31, v23
	.Lbf16w16_s0t1_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w16_s0t1_tile_done:
	s_add_i32 s52, s52, 1
	s_cmp_ge_u32 s52, s43
	s_cbranch_scc1 .Lbf16w16_fold
	s_lshl_b32 s50, s52, 9
	s_lshl_b32 s51, s52, 10
	s_lshl_b32 s49, s52, 8
	s_sub_i32 s49, s30, s49
	s_lshr_b32 s48, s49, 3
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 0x100
	s_cselect_b32 s48, -1, s48
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s51, v3
	v_lshlrev_b32_e32 v1, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_s1t9_wdone
	s_mov_b32 exec_lo, s48
	buffer_load_b128 v[16:19], v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lbf16w16_s1t9_wdone:
	s_add_i32 s45, s41, 0
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w16_s1t0_nofill
	v_add_nc_u32_e32 v13, s60, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w16_s1t0_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_s1t0_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lbf16w16_s1t0_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lbf16w16_s1t0_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v48, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v49, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v50, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v51, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lbf16w16_s1t0_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v52, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v53, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v54, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v55, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lbf16w16_s1t0_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v56, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v57, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v58, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v59, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lbf16w16_s1t0_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v60, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v61, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v62, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v63, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lbf16w16_s1t0_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v64, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v65, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v66, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v67, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v31, v23
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lbf16w16_s1t0_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v68, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v69, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v70, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v71, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v31, v23
	.Lbf16w16_s1t0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w16_s1t0_tile_done:
	s_cmp_le_u32 s31, 8
	s_cbranch_scc1 .Lbf16w16_s1t1_tile_done
	s_add_i32 s45, s41, 8
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w16_s1t1_nofill
	v_add_nc_u32_e32 v13, s61, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w16_s1t1_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_s1t1_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v72, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v73, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v74, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v75, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v72, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v73, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v74, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v75, v31, v23
	s_cmp_le_u32 s31, 9
	s_cbranch_scc1 .Lbf16w16_s1t1_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v76, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v77, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v78, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v79, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v76, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v77, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v78, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v79, v31, v23
	s_cmp_le_u32 s31, 10
	s_cbranch_scc1 .Lbf16w16_s1t1_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v80, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v81, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v82, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v83, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v80, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v81, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v82, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v83, v31, v23
	s_cmp_le_u32 s31, 11
	s_cbranch_scc1 .Lbf16w16_s1t1_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v84, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v85, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v86, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v87, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v84, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v85, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v86, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v87, v31, v23
	s_cmp_le_u32 s31, 12
	s_cbranch_scc1 .Lbf16w16_s1t1_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v88, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v89, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v90, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v91, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s31, 13
	s_cbranch_scc1 .Lbf16w16_s1t1_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v92, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v93, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v94, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v95, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v31, v23
	s_cmp_le_u32 s31, 14
	s_cbranch_scc1 .Lbf16w16_s1t1_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v96, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v97, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v98, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v99, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v31, v23
	s_cmp_le_u32 s31, 15
	s_cbranch_scc1 .Lbf16w16_s1t1_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v100, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v101, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v102, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v103, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v31, v23
	.Lbf16w16_s1t1_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w16_s1t1_tile_done:
	s_add_i32 s52, s52, 1
	s_cmp_ge_u32 s52, s43
	s_cbranch_scc1 .Lbf16w16_fold
	s_lshl_b32 s50, s52, 9
	s_lshl_b32 s51, s52, 10
	s_lshl_b32 s49, s52, 8
	s_sub_i32 s49, s30, s49
	s_lshr_b32 s48, s49, 3
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 0x100
	s_cselect_b32 s48, -1, s48
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s51, v3
	v_lshlrev_b32_e32 v1, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_s2t9_wdone
	s_mov_b32 exec_lo, s48
	buffer_load_b128 v[16:19], v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lbf16w16_s2t9_wdone:
	s_add_i32 s45, s41, 0
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w16_s2t0_nofill
	v_add_nc_u32_e32 v13, s60, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w16_s2t0_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_s2t0_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lbf16w16_s2t0_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lbf16w16_s2t0_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v48, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v49, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v50, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v51, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lbf16w16_s2t0_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v52, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v53, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v54, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v55, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lbf16w16_s2t0_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v56, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v57, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v58, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v59, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lbf16w16_s2t0_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v60, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v61, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v62, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v63, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lbf16w16_s2t0_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v64, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v65, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v66, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v67, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v31, v23
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lbf16w16_s2t0_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v68, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v69, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v70, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v71, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v31, v23
	.Lbf16w16_s2t0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w16_s2t0_tile_done:
	s_cmp_le_u32 s31, 8
	s_cbranch_scc1 .Lbf16w16_s2t1_tile_done
	s_add_i32 s45, s41, 8
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w16_s2t1_nofill
	v_add_nc_u32_e32 v13, s61, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w16_s2t1_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_s2t1_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v72, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v73, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v74, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v75, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v72, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v73, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v74, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v75, v31, v23
	s_cmp_le_u32 s31, 9
	s_cbranch_scc1 .Lbf16w16_s2t1_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v76, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v77, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v78, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v79, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v76, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v77, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v78, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v79, v31, v23
	s_cmp_le_u32 s31, 10
	s_cbranch_scc1 .Lbf16w16_s2t1_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v80, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v81, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v82, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v83, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v80, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v81, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v82, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v83, v31, v23
	s_cmp_le_u32 s31, 11
	s_cbranch_scc1 .Lbf16w16_s2t1_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v84, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v85, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v86, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v87, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v84, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v85, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v86, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v87, v31, v23
	s_cmp_le_u32 s31, 12
	s_cbranch_scc1 .Lbf16w16_s2t1_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v88, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v89, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v90, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v91, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s31, 13
	s_cbranch_scc1 .Lbf16w16_s2t1_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v92, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v93, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v94, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v95, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v31, v23
	s_cmp_le_u32 s31, 14
	s_cbranch_scc1 .Lbf16w16_s2t1_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v96, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v97, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v98, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v99, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v31, v23
	s_cmp_le_u32 s31, 15
	s_cbranch_scc1 .Lbf16w16_s2t1_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v100, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v101, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v102, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v103, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v31, v23
	.Lbf16w16_s2t1_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w16_s2t1_tile_done:
	s_add_i32 s52, s52, 1
	s_cmp_ge_u32 s52, s43
	s_cbranch_scc1 .Lbf16w16_fold
	s_lshl_b32 s50, s52, 9
	s_lshl_b32 s51, s52, 10
	s_lshl_b32 s49, s52, 8
	s_sub_i32 s49, s30, s49
	s_lshr_b32 s48, s49, 3
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 0x100
	s_cselect_b32 s48, -1, s48
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s51, v3
	v_lshlrev_b32_e32 v1, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_s3t9_wdone
	s_mov_b32 exec_lo, s48
	buffer_load_b128 v[16:19], v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lbf16w16_s3t9_wdone:
	s_add_i32 s45, s41, 0
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w16_s3t0_nofill
	v_add_nc_u32_e32 v13, s60, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w16_s3t0_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_s3t0_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lbf16w16_s3t0_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lbf16w16_s3t0_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v48, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v49, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v50, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v51, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lbf16w16_s3t0_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v52, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v53, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v54, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v55, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lbf16w16_s3t0_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v56, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v57, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v58, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v59, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lbf16w16_s3t0_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v60, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v61, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v62, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v63, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lbf16w16_s3t0_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v64, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v65, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v66, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v67, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v31, v23
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lbf16w16_s3t0_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v68, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v69, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v70, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v71, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v31, v23
	.Lbf16w16_s3t0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w16_s3t0_tile_done:
	s_cmp_le_u32 s31, 8
	s_cbranch_scc1 .Lbf16w16_s3t1_tile_done
	s_add_i32 s45, s41, 8
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w16_s3t1_nofill
	v_add_nc_u32_e32 v13, s61, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w16_s3t1_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_s3t1_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v72, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v73, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v74, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v75, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v72, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v73, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v74, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v75, v31, v23
	s_cmp_le_u32 s31, 9
	s_cbranch_scc1 .Lbf16w16_s3t1_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v76, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v77, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v78, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v79, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v76, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v77, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v78, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v79, v31, v23
	s_cmp_le_u32 s31, 10
	s_cbranch_scc1 .Lbf16w16_s3t1_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v80, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v81, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v82, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v83, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v80, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v81, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v82, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v83, v31, v23
	s_cmp_le_u32 s31, 11
	s_cbranch_scc1 .Lbf16w16_s3t1_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v84, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v85, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v86, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v87, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v84, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v85, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v86, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v87, v31, v23
	s_cmp_le_u32 s31, 12
	s_cbranch_scc1 .Lbf16w16_s3t1_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v88, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v89, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v90, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v91, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s31, 13
	s_cbranch_scc1 .Lbf16w16_s3t1_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v92, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v93, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v94, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v95, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v31, v23
	s_cmp_le_u32 s31, 14
	s_cbranch_scc1 .Lbf16w16_s3t1_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v96, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v97, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v98, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v99, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v31, v23
	s_cmp_le_u32 s31, 15
	s_cbranch_scc1 .Lbf16w16_s3t1_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v100, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v101, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v102, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v103, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v31, v23
	.Lbf16w16_s3t1_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w16_s3t1_tile_done:
	s_add_i32 s52, s52, 1
	s_cmp_lt_u32 s52, s43
	s_cbranch_scc1 .Lbf16w16_group
	s_branch .Lbf16w16_fold
	.Lbf16w16_scalar:
	.Lbf16w16_tile32:
	s_lshl_b32 s50, s52, 6
	s_lshl_b32 s51, s52, 7
	s_lshl_b32 s49, s52, 5
	s_sub_i32 s49, s30, s49
	s_mov_b32 s48, s49
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 32
	s_cselect_b32 s48, -1, s48
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s51, v3
	v_lshlrev_b32_e32 v1, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_sct9_wdone
	s_mov_b32 exec_lo, s48
	buffer_load_u16 v16, v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lbf16w16_sct9_wdone:
	s_add_i32 s45, s41, 0
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w16_sct0_nofill
	v_add_nc_u32_e32 v13, s60, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v13, s[16:17]
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v24
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w16_sct0_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_sct0_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v40, v24, v16
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lbf16w16_sct0_cdone
	ds_load_b32 v24, v12 offset:1024
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v44, v24, v16
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lbf16w16_sct0_cdone
	ds_load_b32 v24, v12 offset:2048
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v48, v24, v16
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lbf16w16_sct0_cdone
	ds_load_b32 v24, v12 offset:3072
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v52, v24, v16
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lbf16w16_sct0_cdone
	ds_load_b32 v24, v12 offset:4096
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v56, v24, v16
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lbf16w16_sct0_cdone
	ds_load_b32 v24, v12 offset:5120
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v60, v24, v16
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lbf16w16_sct0_cdone
	ds_load_b32 v24, v12 offset:6144
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v64, v24, v16
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lbf16w16_sct0_cdone
	ds_load_b32 v24, v12 offset:7168
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v68, v24, v16
	.Lbf16w16_sct0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w16_sct0_tile_done:
	s_cmp_le_u32 s31, 8
	s_cbranch_scc1 .Lbf16w16_sct1_tile_done
	s_add_i32 s45, s41, 8
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w16_sct1_nofill
	v_add_nc_u32_e32 v13, s61, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v13, s[16:17]
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v24
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w16_sct1_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_sct1_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v72, v24, v16
	s_cmp_le_u32 s31, 9
	s_cbranch_scc1 .Lbf16w16_sct1_cdone
	ds_load_b32 v24, v12 offset:1024
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v76, v24, v16
	s_cmp_le_u32 s31, 10
	s_cbranch_scc1 .Lbf16w16_sct1_cdone
	ds_load_b32 v24, v12 offset:2048
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v80, v24, v16
	s_cmp_le_u32 s31, 11
	s_cbranch_scc1 .Lbf16w16_sct1_cdone
	ds_load_b32 v24, v12 offset:3072
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v84, v24, v16
	s_cmp_le_u32 s31, 12
	s_cbranch_scc1 .Lbf16w16_sct1_cdone
	ds_load_b32 v24, v12 offset:4096
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v88, v24, v16
	s_cmp_le_u32 s31, 13
	s_cbranch_scc1 .Lbf16w16_sct1_cdone
	ds_load_b32 v24, v12 offset:5120
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v92, v24, v16
	s_cmp_le_u32 s31, 14
	s_cbranch_scc1 .Lbf16w16_sct1_cdone
	ds_load_b32 v24, v12 offset:6144
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v96, v24, v16
	s_cmp_le_u32 s31, 15
	s_cbranch_scc1 .Lbf16w16_sct1_cdone
	ds_load_b32 v24, v12 offset:7168
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v100, v24, v16
	.Lbf16w16_sct1_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w16_sct1_tile_done:
	s_add_i32 s52, s52, 1
	s_cmp_lt_u32 s52, s44
	s_cbranch_scc1 .Lbf16w16_tile32
	.Lbf16w16_fold:
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w16_end
	v_add_f32_e32 v41, v41, v40
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v41
	v_add_f32_e32 v45, v45, v44
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v45
	v_add_f32_e32 v49, v49, v48
	v_add_f32_e32 v51, v51, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v49
	v_add_f32_e32 v53, v53, v52
	v_add_f32_e32 v55, v55, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v53
	v_add_f32_e32 v57, v57, v56
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v57
	v_add_f32_e32 v61, v61, v60
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v61
	v_add_f32_e32 v65, v65, v64
	v_add_f32_e32 v67, v67, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v65
	v_add_f32_e32 v69, v69, v68
	v_add_f32_e32 v71, v71, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v69
	v_add_f32_e32 v73, v73, v72
	v_add_f32_e32 v75, v75, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v75, v75, v73
	v_add_f32_e32 v77, v77, v76
	v_add_f32_e32 v79, v79, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v79, v79, v77
	v_add_f32_e32 v81, v81, v80
	v_add_f32_e32 v83, v83, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v83, v83, v81
	v_add_f32_e32 v85, v85, v84
	v_add_f32_e32 v87, v87, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v87, v87, v85
	v_add_f32_e32 v89, v89, v88
	v_add_f32_e32 v91, v91, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v91, v91, v89
	v_add_f32_e32 v93, v93, v92
	v_add_f32_e32 v95, v95, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v95, v95, v93
	v_add_f32_e32 v97, v97, v96
	v_add_f32_e32 v99, v99, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v99, v99, v97
	v_add_f32_e32 v101, v101, v100
	v_add_f32_e32 v103, v103, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v103, v103, v101
	ds_swizzle_b32 v24, v43 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v47 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v51 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v55 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v59 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v63 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v71 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v75 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v79 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v83 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v87 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v36, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v37, v95 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v38, v99 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v39, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmpx_eq_u32_e32 0, v5
	s_mul_i32 s55, s54, 0
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v24, s55
	s_mul_i32 s55, s54, 1
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v25, s55
	s_mul_i32 s55, s54, 2
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v26, s55
	s_mul_i32 s55, s54, 3
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v27, s55
	s_mul_i32 s55, s54, 4
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v28, s55
	s_mul_i32 s55, s54, 5
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v29, s55
	s_mul_i32 s55, s54, 6
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v30, s55
	s_mul_i32 s55, s54, 7
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v31, s55
	s_mul_i32 s55, s54, 8
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v32, s55
	s_mul_i32 s55, s54, 9
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v33, s55
	s_mul_i32 s55, s54, 10
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v34, s55
	s_mul_i32 s55, s54, 11
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v35, s55
	s_mul_i32 s55, s54, 12
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v36, s55
	s_mul_i32 s55, s54, 13
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v37, s55
	s_mul_i32 s55, s54, 14
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v38, s55
	s_mul_i32 s55, s54, 15
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v39, s55
	s_cmp_le_u32 s31, 0
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v24, v43, s[32:33]
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v25, v47, s[32:33]
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v26, v51, s[32:33]
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v27, v55, s[32:33]
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v28, v59, s[32:33]
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v29, v63, s[32:33]
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v30, v67, s[32:33]
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v31, v71, s[32:33]
	s_cmp_le_u32 s31, 8
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v32, v75, s[32:33]
	s_cmp_le_u32 s31, 9
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v33, v79, s[32:33]
	s_cmp_le_u32 s31, 10
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v34, v83, s[32:33]
	s_cmp_le_u32 s31, 11
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v35, v87, s[32:33]
	s_cmp_le_u32 s31, 12
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v36, v91, s[32:33]
	s_cmp_le_u32 s31, 13
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v37, v95, s[32:33]
	s_cmp_le_u32 s31, 14
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v38, v99, s[32:33]
	s_cmp_le_u32 s31, 15
	s_cbranch_scc1 .Lbf16w16_stored0
	global_store_b32 v39, v103, s[32:33]
	.Lbf16w16_stored0:
	s_waitcnt_vscnt null, 0x0
	.Lbf16w16_end:
	s_endpgm
.Lgemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151_end:
.size gemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151, .Lgemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151_end-gemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151
	.amdhsa_group_segment_fixed_size 8192
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
	.amdhsa_user_sgpr_count 2
	.amdhsa_user_sgpr_dispatch_ptr 0
	.amdhsa_user_sgpr_queue_ptr 0
	.amdhsa_user_sgpr_kernarg_segment_ptr 1
	.amdhsa_user_sgpr_dispatch_id 0
	.amdhsa_user_sgpr_private_segment_size 0
	.amdhsa_wavefront_size32 1
	.amdhsa_uses_dynamic_stack 0
	.amdhsa_enable_private_segment 0
	.amdhsa_system_sgpr_workgroup_id_x 1
	.amdhsa_system_sgpr_workgroup_id_y 0
	.amdhsa_system_sgpr_workgroup_id_z 0
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 104
	.amdhsa_next_free_sgpr 64
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151_end-gemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.text
.protected gemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151
.globl gemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151
.p2align 8
.type gemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151,@function
gemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b128 s[28:31], s[0:1], 0x50
	v_lshrrev_b32_e32 v13, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v13
	v_and_b32_e32 v5, 31, v0
	s_mov_b32 s40, s2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s36, s26, 7
	s_lshr_b32 s36, s36, 3
	s_add_i32 s45, s27, 7
	s_lshr_b32 s45, s45, 3
	s_add_i32 s37, s36, s45
	s_add_i32 s45, s28, 7
	s_lshr_b32 s45, s45, 3
	s_add_i32 s38, s37, s45
	s_add_i32 s45, s29, 7
	s_lshr_b32 s45, s45, 3
	s_add_i32 s39, s38, s45
	s_cmp_ge_u32 s40, s39
	s_cbranch_scc1 .Lbf16w32_end
	s_cmp_eq_u32 s31, 0
	s_cbranch_scc1 .Lbf16w32_end
	s_cmp_gt_u32 s31, 32
	s_cbranch_scc1 .Lbf16w32_end
	s_cmp_lt_u32 s40, s38
	s_cselect_b64 s[4:5], s[12:13], s[14:15]
	s_cselect_b64 s[32:33], s[22:23], s[24:25]
	s_cselect_b32 s34, s28, s29
	s_cselect_b32 s35, s37, s38
	s_cmp_lt_u32 s40, s37
	s_cselect_b64 s[4:5], s[10:11], s[4:5]
	s_cselect_b64 s[32:33], s[20:21], s[32:33]
	s_cselect_b32 s34, s27, s34
	s_cselect_b32 s35, s36, s35
	s_cmp_lt_u32 s40, s36
	s_cselect_b64 s[4:5], s[8:9], s[4:5]
	s_cselect_b64 s[32:33], s[18:19], s[32:33]
	s_cselect_b32 s34, s26, s34
	s_cselect_b32 s35, 0, s35
	s_sub_i32 s42, s40, s35
	s_lshl_b32 s42, s42, 3
	s_add_i32 s42, s42, s41
	s_cmp_lt_i32 s42, s34
	s_cselect_b32 s56, 1, 0
	s_and_b32 s58, s30, 7
	s_add_i32 s43, s30, 0xff
	s_lshr_b32 s43, s43, 8
	s_add_i32 s44, s30, 31
	s_lshr_b32 s44, s44, 5
	s_lshl_b32 s59, s30, 1
	s_mul_i32 s46, s42, s59
	s_mul_hi_u32 s47, s42, s59
	s_add_u32 s4, s4, s46
	s_addc_u32 s5, s5, s47
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_lshl_b32 s53, s30, 2
	s_lshl_b32 s54, s34, 2
	s_lshl_b32 s57, s42, 2
	s_add_i32 s45, s41, 0
	s_mul_i32 s60, s45, s53
	s_add_i32 s45, s41, 8
	s_mul_i32 s61, s45, s53
	s_add_i32 s45, s41, 16
	s_mul_i32 s62, s45, s53
	s_add_i32 s45, s41, 24
	s_mul_i32 s63, s45, s53
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v12, 2, v5
	s_lshl_b32 s45, s41, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v9, s45, v12
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	s_mov_b32 s52, 0
	s_cmp_lg_u32 s58, 0
	s_cbranch_scc1 .Lbf16w32_scalar
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lbf16w32_fold
	.Lbf16w32_group:
	s_lshl_b32 s50, s52, 9
	s_lshl_b32 s51, s52, 10
	s_lshl_b32 s49, s52, 8
	s_sub_i32 s49, s30, s49
	s_lshr_b32 s48, s49, 3
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 0x100
	s_cselect_b32 s48, -1, s48
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s51, v3
	v_lshlrev_b32_e32 v1, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s0t9_wdone
	s_mov_b32 exec_lo, s48
	buffer_load_b128 v[16:19], v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s0t9_wdone:
	s_add_i32 s45, s41, 0
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s0t0_nofill
	v_add_nc_u32_e32 v13, s60, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s0t0_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s0t0_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lbf16w32_s0t0_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lbf16w32_s0t0_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v48, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v49, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v50, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v51, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lbf16w32_s0t0_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v52, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v53, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v54, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v55, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lbf16w32_s0t0_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v56, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v57, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v58, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v59, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lbf16w32_s0t0_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v60, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v61, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v62, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v63, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lbf16w32_s0t0_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v64, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v65, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v66, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v67, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v31, v23
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lbf16w32_s0t0_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v68, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v69, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v70, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v71, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v31, v23
	.Lbf16w32_s0t0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s0t0_tile_done:
	s_cmp_le_u32 s31, 8
	s_cbranch_scc1 .Lbf16w32_s0t1_tile_done
	s_add_i32 s45, s41, 8
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s0t1_nofill
	v_add_nc_u32_e32 v13, s61, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s0t1_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s0t1_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v72, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v73, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v74, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v75, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v72, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v73, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v74, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v75, v31, v23
	s_cmp_le_u32 s31, 9
	s_cbranch_scc1 .Lbf16w32_s0t1_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v76, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v77, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v78, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v79, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v76, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v77, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v78, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v79, v31, v23
	s_cmp_le_u32 s31, 10
	s_cbranch_scc1 .Lbf16w32_s0t1_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v80, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v81, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v82, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v83, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v80, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v81, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v82, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v83, v31, v23
	s_cmp_le_u32 s31, 11
	s_cbranch_scc1 .Lbf16w32_s0t1_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v84, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v85, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v86, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v87, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v84, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v85, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v86, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v87, v31, v23
	s_cmp_le_u32 s31, 12
	s_cbranch_scc1 .Lbf16w32_s0t1_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v88, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v89, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v90, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v91, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s31, 13
	s_cbranch_scc1 .Lbf16w32_s0t1_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v92, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v93, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v94, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v95, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v31, v23
	s_cmp_le_u32 s31, 14
	s_cbranch_scc1 .Lbf16w32_s0t1_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v96, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v97, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v98, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v99, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v31, v23
	s_cmp_le_u32 s31, 15
	s_cbranch_scc1 .Lbf16w32_s0t1_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v100, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v101, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v102, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v103, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v31, v23
	.Lbf16w32_s0t1_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s0t1_tile_done:
	s_cmp_le_u32 s31, 16
	s_cbranch_scc1 .Lbf16w32_s0t2_tile_done
	s_add_i32 s45, s41, 16
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s0t2_nofill
	v_add_nc_u32_e32 v13, s62, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s0t2_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s0t2_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v104, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v105, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v106, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v107, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v104, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v105, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v106, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v107, v31, v23
	s_cmp_le_u32 s31, 17
	s_cbranch_scc1 .Lbf16w32_s0t2_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v108, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v109, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v110, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v111, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v108, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v109, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v110, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v111, v31, v23
	s_cmp_le_u32 s31, 18
	s_cbranch_scc1 .Lbf16w32_s0t2_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v112, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v113, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v114, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v115, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v112, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v113, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v114, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v115, v31, v23
	s_cmp_le_u32 s31, 19
	s_cbranch_scc1 .Lbf16w32_s0t2_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v116, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v117, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v118, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v119, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v116, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v117, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v118, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v119, v31, v23
	s_cmp_le_u32 s31, 20
	s_cbranch_scc1 .Lbf16w32_s0t2_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v120, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v121, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v122, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v123, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v120, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v121, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v122, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v123, v31, v23
	s_cmp_le_u32 s31, 21
	s_cbranch_scc1 .Lbf16w32_s0t2_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v124, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v125, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v126, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v127, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v124, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v125, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v126, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v127, v31, v23
	s_cmp_le_u32 s31, 22
	s_cbranch_scc1 .Lbf16w32_s0t2_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v128, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v129, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v130, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v131, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v128, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v129, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v130, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v131, v31, v23
	s_cmp_le_u32 s31, 23
	s_cbranch_scc1 .Lbf16w32_s0t2_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v132, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v133, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v134, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v135, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v132, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v133, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v134, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v135, v31, v23
	.Lbf16w32_s0t2_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s0t2_tile_done:
	s_cmp_le_u32 s31, 24
	s_cbranch_scc1 .Lbf16w32_s0t3_tile_done
	s_add_i32 s45, s41, 24
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s0t3_nofill
	v_add_nc_u32_e32 v13, s63, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s0t3_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s0t3_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v136, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v137, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v138, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v139, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v136, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v137, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v138, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v139, v31, v23
	s_cmp_le_u32 s31, 25
	s_cbranch_scc1 .Lbf16w32_s0t3_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v140, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v141, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v142, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v143, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v140, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v141, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v142, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v143, v31, v23
	s_cmp_le_u32 s31, 26
	s_cbranch_scc1 .Lbf16w32_s0t3_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v144, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v145, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v146, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v147, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v144, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v145, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v146, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v147, v31, v23
	s_cmp_le_u32 s31, 27
	s_cbranch_scc1 .Lbf16w32_s0t3_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v148, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v149, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v150, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v151, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v148, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v149, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v150, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v151, v31, v23
	s_cmp_le_u32 s31, 28
	s_cbranch_scc1 .Lbf16w32_s0t3_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v152, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v153, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v154, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v155, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v152, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v153, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v154, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v155, v31, v23
	s_cmp_le_u32 s31, 29
	s_cbranch_scc1 .Lbf16w32_s0t3_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v156, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v157, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v158, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v159, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v156, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v157, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v158, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v159, v31, v23
	s_cmp_le_u32 s31, 30
	s_cbranch_scc1 .Lbf16w32_s0t3_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v160, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v161, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v162, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v163, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v160, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v161, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v162, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v163, v31, v23
	s_cmp_le_u32 s31, 31
	s_cbranch_scc1 .Lbf16w32_s0t3_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v164, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v165, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v166, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v167, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v164, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v165, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v166, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v167, v31, v23
	.Lbf16w32_s0t3_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s0t3_tile_done:
	s_add_i32 s52, s52, 1
	s_cmp_ge_u32 s52, s43
	s_cbranch_scc1 .Lbf16w32_fold
	s_lshl_b32 s50, s52, 9
	s_lshl_b32 s51, s52, 10
	s_lshl_b32 s49, s52, 8
	s_sub_i32 s49, s30, s49
	s_lshr_b32 s48, s49, 3
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 0x100
	s_cselect_b32 s48, -1, s48
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s51, v3
	v_lshlrev_b32_e32 v1, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s1t9_wdone
	s_mov_b32 exec_lo, s48
	buffer_load_b128 v[16:19], v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s1t9_wdone:
	s_add_i32 s45, s41, 0
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s1t0_nofill
	v_add_nc_u32_e32 v13, s60, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s1t0_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s1t0_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lbf16w32_s1t0_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lbf16w32_s1t0_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v48, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v49, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v50, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v51, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lbf16w32_s1t0_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v52, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v53, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v54, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v55, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lbf16w32_s1t0_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v56, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v57, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v58, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v59, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lbf16w32_s1t0_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v60, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v61, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v62, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v63, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lbf16w32_s1t0_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v64, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v65, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v66, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v67, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v31, v23
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lbf16w32_s1t0_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v68, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v69, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v70, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v71, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v31, v23
	.Lbf16w32_s1t0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s1t0_tile_done:
	s_cmp_le_u32 s31, 8
	s_cbranch_scc1 .Lbf16w32_s1t1_tile_done
	s_add_i32 s45, s41, 8
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s1t1_nofill
	v_add_nc_u32_e32 v13, s61, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s1t1_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s1t1_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v72, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v73, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v74, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v75, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v72, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v73, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v74, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v75, v31, v23
	s_cmp_le_u32 s31, 9
	s_cbranch_scc1 .Lbf16w32_s1t1_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v76, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v77, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v78, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v79, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v76, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v77, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v78, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v79, v31, v23
	s_cmp_le_u32 s31, 10
	s_cbranch_scc1 .Lbf16w32_s1t1_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v80, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v81, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v82, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v83, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v80, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v81, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v82, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v83, v31, v23
	s_cmp_le_u32 s31, 11
	s_cbranch_scc1 .Lbf16w32_s1t1_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v84, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v85, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v86, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v87, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v84, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v85, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v86, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v87, v31, v23
	s_cmp_le_u32 s31, 12
	s_cbranch_scc1 .Lbf16w32_s1t1_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v88, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v89, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v90, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v91, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s31, 13
	s_cbranch_scc1 .Lbf16w32_s1t1_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v92, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v93, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v94, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v95, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v31, v23
	s_cmp_le_u32 s31, 14
	s_cbranch_scc1 .Lbf16w32_s1t1_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v96, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v97, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v98, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v99, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v31, v23
	s_cmp_le_u32 s31, 15
	s_cbranch_scc1 .Lbf16w32_s1t1_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v100, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v101, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v102, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v103, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v31, v23
	.Lbf16w32_s1t1_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s1t1_tile_done:
	s_cmp_le_u32 s31, 16
	s_cbranch_scc1 .Lbf16w32_s1t2_tile_done
	s_add_i32 s45, s41, 16
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s1t2_nofill
	v_add_nc_u32_e32 v13, s62, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s1t2_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s1t2_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v104, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v105, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v106, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v107, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v104, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v105, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v106, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v107, v31, v23
	s_cmp_le_u32 s31, 17
	s_cbranch_scc1 .Lbf16w32_s1t2_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v108, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v109, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v110, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v111, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v108, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v109, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v110, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v111, v31, v23
	s_cmp_le_u32 s31, 18
	s_cbranch_scc1 .Lbf16w32_s1t2_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v112, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v113, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v114, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v115, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v112, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v113, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v114, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v115, v31, v23
	s_cmp_le_u32 s31, 19
	s_cbranch_scc1 .Lbf16w32_s1t2_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v116, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v117, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v118, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v119, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v116, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v117, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v118, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v119, v31, v23
	s_cmp_le_u32 s31, 20
	s_cbranch_scc1 .Lbf16w32_s1t2_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v120, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v121, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v122, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v123, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v120, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v121, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v122, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v123, v31, v23
	s_cmp_le_u32 s31, 21
	s_cbranch_scc1 .Lbf16w32_s1t2_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v124, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v125, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v126, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v127, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v124, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v125, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v126, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v127, v31, v23
	s_cmp_le_u32 s31, 22
	s_cbranch_scc1 .Lbf16w32_s1t2_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v128, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v129, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v130, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v131, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v128, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v129, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v130, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v131, v31, v23
	s_cmp_le_u32 s31, 23
	s_cbranch_scc1 .Lbf16w32_s1t2_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v132, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v133, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v134, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v135, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v132, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v133, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v134, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v135, v31, v23
	.Lbf16w32_s1t2_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s1t2_tile_done:
	s_cmp_le_u32 s31, 24
	s_cbranch_scc1 .Lbf16w32_s1t3_tile_done
	s_add_i32 s45, s41, 24
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s1t3_nofill
	v_add_nc_u32_e32 v13, s63, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s1t3_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s1t3_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v136, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v137, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v138, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v139, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v136, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v137, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v138, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v139, v31, v23
	s_cmp_le_u32 s31, 25
	s_cbranch_scc1 .Lbf16w32_s1t3_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v140, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v141, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v142, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v143, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v140, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v141, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v142, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v143, v31, v23
	s_cmp_le_u32 s31, 26
	s_cbranch_scc1 .Lbf16w32_s1t3_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v144, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v145, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v146, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v147, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v144, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v145, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v146, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v147, v31, v23
	s_cmp_le_u32 s31, 27
	s_cbranch_scc1 .Lbf16w32_s1t3_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v148, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v149, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v150, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v151, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v148, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v149, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v150, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v151, v31, v23
	s_cmp_le_u32 s31, 28
	s_cbranch_scc1 .Lbf16w32_s1t3_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v152, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v153, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v154, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v155, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v152, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v153, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v154, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v155, v31, v23
	s_cmp_le_u32 s31, 29
	s_cbranch_scc1 .Lbf16w32_s1t3_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v156, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v157, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v158, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v159, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v156, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v157, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v158, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v159, v31, v23
	s_cmp_le_u32 s31, 30
	s_cbranch_scc1 .Lbf16w32_s1t3_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v160, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v161, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v162, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v163, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v160, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v161, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v162, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v163, v31, v23
	s_cmp_le_u32 s31, 31
	s_cbranch_scc1 .Lbf16w32_s1t3_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v164, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v165, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v166, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v167, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v164, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v165, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v166, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v167, v31, v23
	.Lbf16w32_s1t3_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s1t3_tile_done:
	s_add_i32 s52, s52, 1
	s_cmp_ge_u32 s52, s43
	s_cbranch_scc1 .Lbf16w32_fold
	s_lshl_b32 s50, s52, 9
	s_lshl_b32 s51, s52, 10
	s_lshl_b32 s49, s52, 8
	s_sub_i32 s49, s30, s49
	s_lshr_b32 s48, s49, 3
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 0x100
	s_cselect_b32 s48, -1, s48
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s51, v3
	v_lshlrev_b32_e32 v1, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s2t9_wdone
	s_mov_b32 exec_lo, s48
	buffer_load_b128 v[16:19], v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s2t9_wdone:
	s_add_i32 s45, s41, 0
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s2t0_nofill
	v_add_nc_u32_e32 v13, s60, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s2t0_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s2t0_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lbf16w32_s2t0_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lbf16w32_s2t0_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v48, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v49, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v50, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v51, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lbf16w32_s2t0_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v52, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v53, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v54, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v55, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lbf16w32_s2t0_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v56, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v57, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v58, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v59, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lbf16w32_s2t0_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v60, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v61, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v62, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v63, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lbf16w32_s2t0_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v64, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v65, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v66, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v67, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v31, v23
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lbf16w32_s2t0_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v68, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v69, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v70, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v71, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v31, v23
	.Lbf16w32_s2t0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s2t0_tile_done:
	s_cmp_le_u32 s31, 8
	s_cbranch_scc1 .Lbf16w32_s2t1_tile_done
	s_add_i32 s45, s41, 8
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s2t1_nofill
	v_add_nc_u32_e32 v13, s61, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s2t1_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s2t1_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v72, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v73, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v74, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v75, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v72, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v73, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v74, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v75, v31, v23
	s_cmp_le_u32 s31, 9
	s_cbranch_scc1 .Lbf16w32_s2t1_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v76, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v77, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v78, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v79, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v76, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v77, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v78, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v79, v31, v23
	s_cmp_le_u32 s31, 10
	s_cbranch_scc1 .Lbf16w32_s2t1_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v80, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v81, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v82, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v83, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v80, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v81, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v82, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v83, v31, v23
	s_cmp_le_u32 s31, 11
	s_cbranch_scc1 .Lbf16w32_s2t1_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v84, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v85, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v86, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v87, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v84, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v85, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v86, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v87, v31, v23
	s_cmp_le_u32 s31, 12
	s_cbranch_scc1 .Lbf16w32_s2t1_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v88, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v89, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v90, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v91, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s31, 13
	s_cbranch_scc1 .Lbf16w32_s2t1_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v92, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v93, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v94, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v95, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v31, v23
	s_cmp_le_u32 s31, 14
	s_cbranch_scc1 .Lbf16w32_s2t1_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v96, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v97, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v98, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v99, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v31, v23
	s_cmp_le_u32 s31, 15
	s_cbranch_scc1 .Lbf16w32_s2t1_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v100, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v101, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v102, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v103, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v31, v23
	.Lbf16w32_s2t1_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s2t1_tile_done:
	s_cmp_le_u32 s31, 16
	s_cbranch_scc1 .Lbf16w32_s2t2_tile_done
	s_add_i32 s45, s41, 16
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s2t2_nofill
	v_add_nc_u32_e32 v13, s62, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s2t2_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s2t2_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v104, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v105, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v106, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v107, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v104, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v105, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v106, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v107, v31, v23
	s_cmp_le_u32 s31, 17
	s_cbranch_scc1 .Lbf16w32_s2t2_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v108, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v109, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v110, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v111, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v108, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v109, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v110, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v111, v31, v23
	s_cmp_le_u32 s31, 18
	s_cbranch_scc1 .Lbf16w32_s2t2_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v112, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v113, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v114, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v115, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v112, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v113, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v114, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v115, v31, v23
	s_cmp_le_u32 s31, 19
	s_cbranch_scc1 .Lbf16w32_s2t2_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v116, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v117, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v118, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v119, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v116, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v117, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v118, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v119, v31, v23
	s_cmp_le_u32 s31, 20
	s_cbranch_scc1 .Lbf16w32_s2t2_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v120, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v121, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v122, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v123, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v120, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v121, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v122, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v123, v31, v23
	s_cmp_le_u32 s31, 21
	s_cbranch_scc1 .Lbf16w32_s2t2_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v124, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v125, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v126, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v127, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v124, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v125, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v126, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v127, v31, v23
	s_cmp_le_u32 s31, 22
	s_cbranch_scc1 .Lbf16w32_s2t2_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v128, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v129, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v130, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v131, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v128, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v129, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v130, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v131, v31, v23
	s_cmp_le_u32 s31, 23
	s_cbranch_scc1 .Lbf16w32_s2t2_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v132, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v133, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v134, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v135, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v132, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v133, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v134, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v135, v31, v23
	.Lbf16w32_s2t2_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s2t2_tile_done:
	s_cmp_le_u32 s31, 24
	s_cbranch_scc1 .Lbf16w32_s2t3_tile_done
	s_add_i32 s45, s41, 24
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s2t3_nofill
	v_add_nc_u32_e32 v13, s63, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s2t3_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s2t3_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v136, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v137, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v138, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v139, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v136, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v137, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v138, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v139, v31, v23
	s_cmp_le_u32 s31, 25
	s_cbranch_scc1 .Lbf16w32_s2t3_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v140, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v141, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v142, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v143, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v140, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v141, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v142, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v143, v31, v23
	s_cmp_le_u32 s31, 26
	s_cbranch_scc1 .Lbf16w32_s2t3_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v144, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v145, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v146, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v147, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v144, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v145, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v146, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v147, v31, v23
	s_cmp_le_u32 s31, 27
	s_cbranch_scc1 .Lbf16w32_s2t3_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v148, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v149, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v150, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v151, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v148, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v149, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v150, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v151, v31, v23
	s_cmp_le_u32 s31, 28
	s_cbranch_scc1 .Lbf16w32_s2t3_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v152, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v153, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v154, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v155, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v152, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v153, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v154, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v155, v31, v23
	s_cmp_le_u32 s31, 29
	s_cbranch_scc1 .Lbf16w32_s2t3_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v156, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v157, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v158, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v159, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v156, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v157, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v158, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v159, v31, v23
	s_cmp_le_u32 s31, 30
	s_cbranch_scc1 .Lbf16w32_s2t3_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v160, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v161, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v162, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v163, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v160, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v161, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v162, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v163, v31, v23
	s_cmp_le_u32 s31, 31
	s_cbranch_scc1 .Lbf16w32_s2t3_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v164, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v165, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v166, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v167, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v164, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v165, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v166, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v167, v31, v23
	.Lbf16w32_s2t3_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s2t3_tile_done:
	s_add_i32 s52, s52, 1
	s_cmp_ge_u32 s52, s43
	s_cbranch_scc1 .Lbf16w32_fold
	s_lshl_b32 s50, s52, 9
	s_lshl_b32 s51, s52, 10
	s_lshl_b32 s49, s52, 8
	s_sub_i32 s49, s30, s49
	s_lshr_b32 s48, s49, 3
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 0x100
	s_cselect_b32 s48, -1, s48
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s51, v3
	v_lshlrev_b32_e32 v1, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s3t9_wdone
	s_mov_b32 exec_lo, s48
	buffer_load_b128 v[16:19], v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s3t9_wdone:
	s_add_i32 s45, s41, 0
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s3t0_nofill
	v_add_nc_u32_e32 v13, s60, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s3t0_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s3t0_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lbf16w32_s3t0_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lbf16w32_s3t0_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v48, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v49, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v50, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v51, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lbf16w32_s3t0_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v52, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v53, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v54, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v55, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lbf16w32_s3t0_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v56, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v57, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v58, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v59, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lbf16w32_s3t0_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v60, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v61, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v62, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v63, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lbf16w32_s3t0_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v64, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v65, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v66, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v67, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v31, v23
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lbf16w32_s3t0_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v68, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v69, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v70, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v71, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v31, v23
	.Lbf16w32_s3t0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s3t0_tile_done:
	s_cmp_le_u32 s31, 8
	s_cbranch_scc1 .Lbf16w32_s3t1_tile_done
	s_add_i32 s45, s41, 8
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s3t1_nofill
	v_add_nc_u32_e32 v13, s61, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s3t1_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s3t1_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v72, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v73, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v74, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v75, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v72, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v73, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v74, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v75, v31, v23
	s_cmp_le_u32 s31, 9
	s_cbranch_scc1 .Lbf16w32_s3t1_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v76, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v77, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v78, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v79, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v76, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v77, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v78, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v79, v31, v23
	s_cmp_le_u32 s31, 10
	s_cbranch_scc1 .Lbf16w32_s3t1_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v80, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v81, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v82, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v83, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v80, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v81, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v82, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v83, v31, v23
	s_cmp_le_u32 s31, 11
	s_cbranch_scc1 .Lbf16w32_s3t1_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v84, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v85, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v86, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v87, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v84, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v85, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v86, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v87, v31, v23
	s_cmp_le_u32 s31, 12
	s_cbranch_scc1 .Lbf16w32_s3t1_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v88, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v89, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v90, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v91, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s31, 13
	s_cbranch_scc1 .Lbf16w32_s3t1_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v92, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v93, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v94, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v95, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v31, v23
	s_cmp_le_u32 s31, 14
	s_cbranch_scc1 .Lbf16w32_s3t1_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v96, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v97, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v98, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v99, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v31, v23
	s_cmp_le_u32 s31, 15
	s_cbranch_scc1 .Lbf16w32_s3t1_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v100, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v101, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v102, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v103, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v31, v23
	.Lbf16w32_s3t1_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s3t1_tile_done:
	s_cmp_le_u32 s31, 16
	s_cbranch_scc1 .Lbf16w32_s3t2_tile_done
	s_add_i32 s45, s41, 16
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s3t2_nofill
	v_add_nc_u32_e32 v13, s62, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s3t2_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s3t2_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v104, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v105, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v106, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v107, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v104, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v105, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v106, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v107, v31, v23
	s_cmp_le_u32 s31, 17
	s_cbranch_scc1 .Lbf16w32_s3t2_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v108, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v109, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v110, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v111, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v108, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v109, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v110, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v111, v31, v23
	s_cmp_le_u32 s31, 18
	s_cbranch_scc1 .Lbf16w32_s3t2_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v112, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v113, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v114, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v115, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v112, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v113, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v114, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v115, v31, v23
	s_cmp_le_u32 s31, 19
	s_cbranch_scc1 .Lbf16w32_s3t2_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v116, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v117, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v118, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v119, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v116, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v117, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v118, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v119, v31, v23
	s_cmp_le_u32 s31, 20
	s_cbranch_scc1 .Lbf16w32_s3t2_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v120, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v121, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v122, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v123, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v120, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v121, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v122, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v123, v31, v23
	s_cmp_le_u32 s31, 21
	s_cbranch_scc1 .Lbf16w32_s3t2_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v124, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v125, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v126, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v127, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v124, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v125, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v126, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v127, v31, v23
	s_cmp_le_u32 s31, 22
	s_cbranch_scc1 .Lbf16w32_s3t2_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v128, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v129, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v130, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v131, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v128, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v129, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v130, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v131, v31, v23
	s_cmp_le_u32 s31, 23
	s_cbranch_scc1 .Lbf16w32_s3t2_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v132, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v133, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v134, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v135, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v132, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v133, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v134, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v135, v31, v23
	.Lbf16w32_s3t2_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s3t2_tile_done:
	s_cmp_le_u32 s31, 24
	s_cbranch_scc1 .Lbf16w32_s3t3_tile_done
	s_add_i32 s45, s41, 24
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_s3t3_nofill
	v_add_nc_u32_e32 v13, s63, v3
	s_mov_b32 exec_lo, s48
	global_load_b128 v[24:27], v13, s[16:17]
	global_load_b128 v[28:31], v13, s[16:17] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_s3t3_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_s3t3_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	ds_load_b32 v25, v12 offset:128
	ds_load_b32 v26, v12 offset:256
	ds_load_b32 v27, v12 offset:384
	ds_load_b32 v28, v12 offset:512
	ds_load_b32 v29, v12 offset:640
	ds_load_b32 v30, v12 offset:768
	ds_load_b32 v31, v12 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v136, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v137, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v138, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v139, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v136, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v137, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v138, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v139, v31, v23
	s_cmp_le_u32 s31, 25
	s_cbranch_scc1 .Lbf16w32_s3t3_cdone
	ds_load_b32 v24, v12 offset:1024
	ds_load_b32 v25, v12 offset:1152
	ds_load_b32 v26, v12 offset:1280
	ds_load_b32 v27, v12 offset:1408
	ds_load_b32 v28, v12 offset:1536
	ds_load_b32 v29, v12 offset:1664
	ds_load_b32 v30, v12 offset:1792
	ds_load_b32 v31, v12 offset:1920
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v140, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v141, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v142, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v143, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v140, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v141, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v142, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v143, v31, v23
	s_cmp_le_u32 s31, 26
	s_cbranch_scc1 .Lbf16w32_s3t3_cdone
	ds_load_b32 v24, v12 offset:2048
	ds_load_b32 v25, v12 offset:2176
	ds_load_b32 v26, v12 offset:2304
	ds_load_b32 v27, v12 offset:2432
	ds_load_b32 v28, v12 offset:2560
	ds_load_b32 v29, v12 offset:2688
	ds_load_b32 v30, v12 offset:2816
	ds_load_b32 v31, v12 offset:2944
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v144, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v145, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v146, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v147, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v144, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v145, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v146, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v147, v31, v23
	s_cmp_le_u32 s31, 27
	s_cbranch_scc1 .Lbf16w32_s3t3_cdone
	ds_load_b32 v24, v12 offset:3072
	ds_load_b32 v25, v12 offset:3200
	ds_load_b32 v26, v12 offset:3328
	ds_load_b32 v27, v12 offset:3456
	ds_load_b32 v28, v12 offset:3584
	ds_load_b32 v29, v12 offset:3712
	ds_load_b32 v30, v12 offset:3840
	ds_load_b32 v31, v12 offset:3968
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v148, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v149, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v150, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v151, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v148, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v149, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v150, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v151, v31, v23
	s_cmp_le_u32 s31, 28
	s_cbranch_scc1 .Lbf16w32_s3t3_cdone
	ds_load_b32 v24, v12 offset:4096
	ds_load_b32 v25, v12 offset:4224
	ds_load_b32 v26, v12 offset:4352
	ds_load_b32 v27, v12 offset:4480
	ds_load_b32 v28, v12 offset:4608
	ds_load_b32 v29, v12 offset:4736
	ds_load_b32 v30, v12 offset:4864
	ds_load_b32 v31, v12 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v152, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v153, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v154, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v155, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v152, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v153, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v154, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v155, v31, v23
	s_cmp_le_u32 s31, 29
	s_cbranch_scc1 .Lbf16w32_s3t3_cdone
	ds_load_b32 v24, v12 offset:5120
	ds_load_b32 v25, v12 offset:5248
	ds_load_b32 v26, v12 offset:5376
	ds_load_b32 v27, v12 offset:5504
	ds_load_b32 v28, v12 offset:5632
	ds_load_b32 v29, v12 offset:5760
	ds_load_b32 v30, v12 offset:5888
	ds_load_b32 v31, v12 offset:6016
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v156, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v157, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v158, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v159, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v156, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v157, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v158, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v159, v31, v23
	s_cmp_le_u32 s31, 30
	s_cbranch_scc1 .Lbf16w32_s3t3_cdone
	ds_load_b32 v24, v12 offset:6144
	ds_load_b32 v25, v12 offset:6272
	ds_load_b32 v26, v12 offset:6400
	ds_load_b32 v27, v12 offset:6528
	ds_load_b32 v28, v12 offset:6656
	ds_load_b32 v29, v12 offset:6784
	ds_load_b32 v30, v12 offset:6912
	ds_load_b32 v31, v12 offset:7040
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v160, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v161, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v162, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v163, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v160, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v161, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v162, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v163, v31, v23
	s_cmp_le_u32 s31, 31
	s_cbranch_scc1 .Lbf16w32_s3t3_cdone
	ds_load_b32 v24, v12 offset:7168
	ds_load_b32 v25, v12 offset:7296
	ds_load_b32 v26, v12 offset:7424
	ds_load_b32 v27, v12 offset:7552
	ds_load_b32 v28, v12 offset:7680
	ds_load_b32 v29, v12 offset:7808
	ds_load_b32 v30, v12 offset:7936
	ds_load_b32 v31, v12 offset:8064
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v164, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v165, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v166, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v167, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v164, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v165, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v166, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v167, v31, v23
	.Lbf16w32_s3t3_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_s3t3_tile_done:
	s_add_i32 s52, s52, 1
	s_cmp_lt_u32 s52, s43
	s_cbranch_scc1 .Lbf16w32_group
	s_branch .Lbf16w32_fold
	.Lbf16w32_scalar:
	.Lbf16w32_tile32:
	s_lshl_b32 s50, s52, 6
	s_lshl_b32 s51, s52, 7
	s_lshl_b32 s49, s52, 5
	s_sub_i32 s49, s30, s49
	s_mov_b32 s48, s49
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 32
	s_cselect_b32 s48, -1, s48
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s51, v3
	v_lshlrev_b32_e32 v1, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_sct9_wdone
	s_mov_b32 exec_lo, s48
	buffer_load_u16 v16, v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lbf16w32_sct9_wdone:
	s_add_i32 s45, s41, 0
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_sct0_nofill
	v_add_nc_u32_e32 v13, s60, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v13, s[16:17]
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v24
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_sct0_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_sct0_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v40, v24, v16
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lbf16w32_sct0_cdone
	ds_load_b32 v24, v12 offset:1024
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v44, v24, v16
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lbf16w32_sct0_cdone
	ds_load_b32 v24, v12 offset:2048
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v48, v24, v16
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lbf16w32_sct0_cdone
	ds_load_b32 v24, v12 offset:3072
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v52, v24, v16
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lbf16w32_sct0_cdone
	ds_load_b32 v24, v12 offset:4096
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v56, v24, v16
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lbf16w32_sct0_cdone
	ds_load_b32 v24, v12 offset:5120
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v60, v24, v16
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lbf16w32_sct0_cdone
	ds_load_b32 v24, v12 offset:6144
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v64, v24, v16
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lbf16w32_sct0_cdone
	ds_load_b32 v24, v12 offset:7168
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v68, v24, v16
	.Lbf16w32_sct0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_sct0_tile_done:
	s_cmp_le_u32 s31, 8
	s_cbranch_scc1 .Lbf16w32_sct1_tile_done
	s_add_i32 s45, s41, 8
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_sct1_nofill
	v_add_nc_u32_e32 v13, s61, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v13, s[16:17]
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v24
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_sct1_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_sct1_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v72, v24, v16
	s_cmp_le_u32 s31, 9
	s_cbranch_scc1 .Lbf16w32_sct1_cdone
	ds_load_b32 v24, v12 offset:1024
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v76, v24, v16
	s_cmp_le_u32 s31, 10
	s_cbranch_scc1 .Lbf16w32_sct1_cdone
	ds_load_b32 v24, v12 offset:2048
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v80, v24, v16
	s_cmp_le_u32 s31, 11
	s_cbranch_scc1 .Lbf16w32_sct1_cdone
	ds_load_b32 v24, v12 offset:3072
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v84, v24, v16
	s_cmp_le_u32 s31, 12
	s_cbranch_scc1 .Lbf16w32_sct1_cdone
	ds_load_b32 v24, v12 offset:4096
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v88, v24, v16
	s_cmp_le_u32 s31, 13
	s_cbranch_scc1 .Lbf16w32_sct1_cdone
	ds_load_b32 v24, v12 offset:5120
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v92, v24, v16
	s_cmp_le_u32 s31, 14
	s_cbranch_scc1 .Lbf16w32_sct1_cdone
	ds_load_b32 v24, v12 offset:6144
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v96, v24, v16
	s_cmp_le_u32 s31, 15
	s_cbranch_scc1 .Lbf16w32_sct1_cdone
	ds_load_b32 v24, v12 offset:7168
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v100, v24, v16
	.Lbf16w32_sct1_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_sct1_tile_done:
	s_cmp_le_u32 s31, 16
	s_cbranch_scc1 .Lbf16w32_sct2_tile_done
	s_add_i32 s45, s41, 16
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_sct2_nofill
	v_add_nc_u32_e32 v13, s62, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v13, s[16:17]
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v24
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_sct2_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_sct2_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v104, v24, v16
	s_cmp_le_u32 s31, 17
	s_cbranch_scc1 .Lbf16w32_sct2_cdone
	ds_load_b32 v24, v12 offset:1024
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v108, v24, v16
	s_cmp_le_u32 s31, 18
	s_cbranch_scc1 .Lbf16w32_sct2_cdone
	ds_load_b32 v24, v12 offset:2048
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v112, v24, v16
	s_cmp_le_u32 s31, 19
	s_cbranch_scc1 .Lbf16w32_sct2_cdone
	ds_load_b32 v24, v12 offset:3072
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v116, v24, v16
	s_cmp_le_u32 s31, 20
	s_cbranch_scc1 .Lbf16w32_sct2_cdone
	ds_load_b32 v24, v12 offset:4096
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v120, v24, v16
	s_cmp_le_u32 s31, 21
	s_cbranch_scc1 .Lbf16w32_sct2_cdone
	ds_load_b32 v24, v12 offset:5120
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v124, v24, v16
	s_cmp_le_u32 s31, 22
	s_cbranch_scc1 .Lbf16w32_sct2_cdone
	ds_load_b32 v24, v12 offset:6144
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v128, v24, v16
	s_cmp_le_u32 s31, 23
	s_cbranch_scc1 .Lbf16w32_sct2_cdone
	ds_load_b32 v24, v12 offset:7168
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v132, v24, v16
	.Lbf16w32_sct2_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_sct2_tile_done:
	s_cmp_le_u32 s31, 24
	s_cbranch_scc1 .Lbf16w32_sct3_tile_done
	s_add_i32 s45, s41, 24
	s_cmp_ge_u32 s45, s31
	s_cbranch_scc1 .Lbf16w32_sct3_nofill
	v_add_nc_u32_e32 v13, s63, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v13, s[16:17]
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v24
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16w32_sct3_nofill:
	s_barrier
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_sct3_cdone
	s_mov_b32 exec_lo, s48
	ds_load_b32 v24, v12
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v136, v24, v16
	s_cmp_le_u32 s31, 25
	s_cbranch_scc1 .Lbf16w32_sct3_cdone
	ds_load_b32 v24, v12 offset:1024
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v140, v24, v16
	s_cmp_le_u32 s31, 26
	s_cbranch_scc1 .Lbf16w32_sct3_cdone
	ds_load_b32 v24, v12 offset:2048
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v144, v24, v16
	s_cmp_le_u32 s31, 27
	s_cbranch_scc1 .Lbf16w32_sct3_cdone
	ds_load_b32 v24, v12 offset:3072
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v148, v24, v16
	s_cmp_le_u32 s31, 28
	s_cbranch_scc1 .Lbf16w32_sct3_cdone
	ds_load_b32 v24, v12 offset:4096
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v152, v24, v16
	s_cmp_le_u32 s31, 29
	s_cbranch_scc1 .Lbf16w32_sct3_cdone
	ds_load_b32 v24, v12 offset:5120
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v156, v24, v16
	s_cmp_le_u32 s31, 30
	s_cbranch_scc1 .Lbf16w32_sct3_cdone
	ds_load_b32 v24, v12 offset:6144
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v160, v24, v16
	s_cmp_le_u32 s31, 31
	s_cbranch_scc1 .Lbf16w32_sct3_cdone
	ds_load_b32 v24, v12 offset:7168
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v164, v24, v16
	.Lbf16w32_sct3_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16w32_sct3_tile_done:
	s_add_i32 s52, s52, 1
	s_cmp_lt_u32 s52, s44
	s_cbranch_scc1 .Lbf16w32_tile32
	.Lbf16w32_fold:
	s_cmp_eq_u32 s56, 0
	s_cbranch_scc1 .Lbf16w32_end
	v_add_f32_e32 v41, v41, v40
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v41
	v_add_f32_e32 v45, v45, v44
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v45
	v_add_f32_e32 v49, v49, v48
	v_add_f32_e32 v51, v51, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v49
	v_add_f32_e32 v53, v53, v52
	v_add_f32_e32 v55, v55, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v53
	v_add_f32_e32 v57, v57, v56
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v57
	v_add_f32_e32 v61, v61, v60
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v61
	v_add_f32_e32 v65, v65, v64
	v_add_f32_e32 v67, v67, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v65
	v_add_f32_e32 v69, v69, v68
	v_add_f32_e32 v71, v71, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v69
	v_add_f32_e32 v73, v73, v72
	v_add_f32_e32 v75, v75, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v75, v75, v73
	v_add_f32_e32 v77, v77, v76
	v_add_f32_e32 v79, v79, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v79, v79, v77
	v_add_f32_e32 v81, v81, v80
	v_add_f32_e32 v83, v83, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v83, v83, v81
	v_add_f32_e32 v85, v85, v84
	v_add_f32_e32 v87, v87, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v87, v87, v85
	v_add_f32_e32 v89, v89, v88
	v_add_f32_e32 v91, v91, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v91, v91, v89
	v_add_f32_e32 v93, v93, v92
	v_add_f32_e32 v95, v95, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v95, v95, v93
	v_add_f32_e32 v97, v97, v96
	v_add_f32_e32 v99, v99, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v99, v99, v97
	v_add_f32_e32 v101, v101, v100
	v_add_f32_e32 v103, v103, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v103, v103, v101
	v_add_f32_e32 v105, v105, v104
	v_add_f32_e32 v107, v107, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v107, v107, v105
	v_add_f32_e32 v109, v109, v108
	v_add_f32_e32 v111, v111, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v111, v111, v109
	v_add_f32_e32 v113, v113, v112
	v_add_f32_e32 v115, v115, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v115, v115, v113
	v_add_f32_e32 v117, v117, v116
	v_add_f32_e32 v119, v119, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v119, v119, v117
	v_add_f32_e32 v121, v121, v120
	v_add_f32_e32 v123, v123, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v123, v123, v121
	v_add_f32_e32 v125, v125, v124
	v_add_f32_e32 v127, v127, v126
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v127, v127, v125
	v_add_f32_e32 v129, v129, v128
	v_add_f32_e32 v131, v131, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v131, v131, v129
	v_add_f32_e32 v133, v133, v132
	v_add_f32_e32 v135, v135, v134
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v135, v135, v133
	v_add_f32_e32 v137, v137, v136
	v_add_f32_e32 v139, v139, v138
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v139, v139, v137
	v_add_f32_e32 v141, v141, v140
	v_add_f32_e32 v143, v143, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v143, v143, v141
	v_add_f32_e32 v145, v145, v144
	v_add_f32_e32 v147, v147, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v147, v147, v145
	v_add_f32_e32 v149, v149, v148
	v_add_f32_e32 v151, v151, v150
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v151, v151, v149
	v_add_f32_e32 v153, v153, v152
	v_add_f32_e32 v155, v155, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v155, v155, v153
	v_add_f32_e32 v157, v157, v156
	v_add_f32_e32 v159, v159, v158
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v159, v159, v157
	v_add_f32_e32 v161, v161, v160
	v_add_f32_e32 v163, v163, v162
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v163, v163, v161
	v_add_f32_e32 v165, v165, v164
	v_add_f32_e32 v167, v167, v166
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v167, v167, v165
	ds_swizzle_b32 v24, v43 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v47 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v51 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v55 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v59 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v63 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v71 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v75 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v79 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v83 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v87 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v36, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v37, v95 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v38, v99 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v39, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	ds_swizzle_b32 v24, v107 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v111 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v119 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v123 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v127 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v131 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v135 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v139 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v143 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v147 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v151 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v36, v155 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v37, v159 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v38, v163 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v39, v167 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v107, v107, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v111, v111, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v115, v115, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v119, v119, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v123, v123, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v127, v127, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v131, v131, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v135, v135, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v139, v139, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v143, v143, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v147, v147, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v151, v151, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v155, v155, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v159, v159, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v163, v163, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v167, v167, v39
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v107
	ds_bpermute_b32 v25, v4, v111
	ds_bpermute_b32 v26, v4, v115
	ds_bpermute_b32 v27, v4, v119
	ds_bpermute_b32 v28, v4, v123
	ds_bpermute_b32 v29, v4, v127
	ds_bpermute_b32 v30, v4, v131
	ds_bpermute_b32 v31, v4, v135
	ds_bpermute_b32 v32, v4, v139
	ds_bpermute_b32 v33, v4, v143
	ds_bpermute_b32 v34, v4, v147
	ds_bpermute_b32 v35, v4, v151
	ds_bpermute_b32 v36, v4, v155
	ds_bpermute_b32 v37, v4, v159
	ds_bpermute_b32 v38, v4, v163
	ds_bpermute_b32 v39, v4, v167
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v107, v107, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v111, v111, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v115, v115, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v119, v119, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v123, v123, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v127, v127, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v131, v131, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v135, v135, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v139, v139, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v143, v143, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v147, v147, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v151, v151, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v155, v155, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v159, v159, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v163, v163, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v167, v167, v39
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v107
	ds_bpermute_b32 v25, v4, v111
	ds_bpermute_b32 v26, v4, v115
	ds_bpermute_b32 v27, v4, v119
	ds_bpermute_b32 v28, v4, v123
	ds_bpermute_b32 v29, v4, v127
	ds_bpermute_b32 v30, v4, v131
	ds_bpermute_b32 v31, v4, v135
	ds_bpermute_b32 v32, v4, v139
	ds_bpermute_b32 v33, v4, v143
	ds_bpermute_b32 v34, v4, v147
	ds_bpermute_b32 v35, v4, v151
	ds_bpermute_b32 v36, v4, v155
	ds_bpermute_b32 v37, v4, v159
	ds_bpermute_b32 v38, v4, v163
	ds_bpermute_b32 v39, v4, v167
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v107, v107, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v111, v111, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v115, v115, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v119, v119, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v123, v123, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v127, v127, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v131, v131, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v135, v135, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v139, v139, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v143, v143, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v147, v147, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v151, v151, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v155, v155, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v159, v159, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v163, v163, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v167, v167, v39
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v107
	ds_bpermute_b32 v25, v4, v111
	ds_bpermute_b32 v26, v4, v115
	ds_bpermute_b32 v27, v4, v119
	ds_bpermute_b32 v28, v4, v123
	ds_bpermute_b32 v29, v4, v127
	ds_bpermute_b32 v30, v4, v131
	ds_bpermute_b32 v31, v4, v135
	ds_bpermute_b32 v32, v4, v139
	ds_bpermute_b32 v33, v4, v143
	ds_bpermute_b32 v34, v4, v147
	ds_bpermute_b32 v35, v4, v151
	ds_bpermute_b32 v36, v4, v155
	ds_bpermute_b32 v37, v4, v159
	ds_bpermute_b32 v38, v4, v163
	ds_bpermute_b32 v39, v4, v167
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v107, v107, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v111, v111, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v115, v115, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v119, v119, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v123, v123, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v127, v127, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v131, v131, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v135, v135, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v139, v139, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v143, v143, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v147, v147, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v151, v151, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v155, v155, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v159, v159, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v163, v163, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v167, v167, v39
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v107
	ds_bpermute_b32 v25, v4, v111
	ds_bpermute_b32 v26, v4, v115
	ds_bpermute_b32 v27, v4, v119
	ds_bpermute_b32 v28, v4, v123
	ds_bpermute_b32 v29, v4, v127
	ds_bpermute_b32 v30, v4, v131
	ds_bpermute_b32 v31, v4, v135
	ds_bpermute_b32 v32, v4, v139
	ds_bpermute_b32 v33, v4, v143
	ds_bpermute_b32 v34, v4, v147
	ds_bpermute_b32 v35, v4, v151
	ds_bpermute_b32 v36, v4, v155
	ds_bpermute_b32 v37, v4, v159
	ds_bpermute_b32 v38, v4, v163
	ds_bpermute_b32 v39, v4, v167
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v107, v107, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v111, v111, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v115, v115, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v119, v119, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v123, v123, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v127, v127, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v131, v131, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v135, v135, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v139, v139, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v143, v143, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v147, v147, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v151, v151, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v155, v155, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v159, v159, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v163, v163, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v167, v167, v39
	v_cmpx_eq_u32_e32 0, v5
	s_mul_i32 s55, s54, 0
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v24, s55
	s_mul_i32 s55, s54, 1
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v25, s55
	s_mul_i32 s55, s54, 2
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v26, s55
	s_mul_i32 s55, s54, 3
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v27, s55
	s_mul_i32 s55, s54, 4
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v28, s55
	s_mul_i32 s55, s54, 5
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v29, s55
	s_mul_i32 s55, s54, 6
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v30, s55
	s_mul_i32 s55, s54, 7
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v31, s55
	s_mul_i32 s55, s54, 8
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v32, s55
	s_mul_i32 s55, s54, 9
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v33, s55
	s_mul_i32 s55, s54, 10
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v34, s55
	s_mul_i32 s55, s54, 11
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v35, s55
	s_mul_i32 s55, s54, 12
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v36, s55
	s_mul_i32 s55, s54, 13
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v37, s55
	s_mul_i32 s55, s54, 14
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v38, s55
	s_mul_i32 s55, s54, 15
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v39, s55
	s_cmp_le_u32 s31, 0
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v24, v43, s[32:33]
	s_cmp_le_u32 s31, 1
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v25, v47, s[32:33]
	s_cmp_le_u32 s31, 2
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v26, v51, s[32:33]
	s_cmp_le_u32 s31, 3
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v27, v55, s[32:33]
	s_cmp_le_u32 s31, 4
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v28, v59, s[32:33]
	s_cmp_le_u32 s31, 5
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v29, v63, s[32:33]
	s_cmp_le_u32 s31, 6
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v30, v67, s[32:33]
	s_cmp_le_u32 s31, 7
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v31, v71, s[32:33]
	s_cmp_le_u32 s31, 8
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v32, v75, s[32:33]
	s_cmp_le_u32 s31, 9
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v33, v79, s[32:33]
	s_cmp_le_u32 s31, 10
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v34, v83, s[32:33]
	s_cmp_le_u32 s31, 11
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v35, v87, s[32:33]
	s_cmp_le_u32 s31, 12
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v36, v91, s[32:33]
	s_cmp_le_u32 s31, 13
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v37, v95, s[32:33]
	s_cmp_le_u32 s31, 14
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v38, v99, s[32:33]
	s_cmp_le_u32 s31, 15
	s_cbranch_scc1 .Lbf16w32_stored0
	global_store_b32 v39, v103, s[32:33]
	.Lbf16w32_stored0:
	s_waitcnt_vscnt null, 0x0
	s_cmp_le_u32 s31, 16
	s_cbranch_scc1 .Lbf16w32_end
	s_mul_i32 s55, s54, 16
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v24, s55
	s_mul_i32 s55, s54, 17
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v25, s55
	s_mul_i32 s55, s54, 18
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v26, s55
	s_mul_i32 s55, s54, 19
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v27, s55
	s_mul_i32 s55, s54, 20
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v28, s55
	s_mul_i32 s55, s54, 21
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v29, s55
	s_mul_i32 s55, s54, 22
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v30, s55
	s_mul_i32 s55, s54, 23
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v31, s55
	s_mul_i32 s55, s54, 24
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v32, s55
	s_mul_i32 s55, s54, 25
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v33, s55
	s_mul_i32 s55, s54, 26
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v34, s55
	s_mul_i32 s55, s54, 27
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v35, s55
	s_mul_i32 s55, s54, 28
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v36, s55
	s_mul_i32 s55, s54, 29
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v37, s55
	s_mul_i32 s55, s54, 30
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v38, s55
	s_mul_i32 s55, s54, 31
	s_add_i32 s55, s55, s57
	v_mov_b32_e32 v39, s55
	s_cmp_le_u32 s31, 16
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v24, v107, s[32:33]
	s_cmp_le_u32 s31, 17
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v25, v111, s[32:33]
	s_cmp_le_u32 s31, 18
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v26, v115, s[32:33]
	s_cmp_le_u32 s31, 19
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v27, v119, s[32:33]
	s_cmp_le_u32 s31, 20
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v28, v123, s[32:33]
	s_cmp_le_u32 s31, 21
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v29, v127, s[32:33]
	s_cmp_le_u32 s31, 22
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v30, v131, s[32:33]
	s_cmp_le_u32 s31, 23
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v31, v135, s[32:33]
	s_cmp_le_u32 s31, 24
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v32, v139, s[32:33]
	s_cmp_le_u32 s31, 25
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v33, v143, s[32:33]
	s_cmp_le_u32 s31, 26
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v34, v147, s[32:33]
	s_cmp_le_u32 s31, 27
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v35, v151, s[32:33]
	s_cmp_le_u32 s31, 28
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v36, v155, s[32:33]
	s_cmp_le_u32 s31, 29
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v37, v159, s[32:33]
	s_cmp_le_u32 s31, 30
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v38, v163, s[32:33]
	s_cmp_le_u32 s31, 31
	s_cbranch_scc1 .Lbf16w32_stored1
	global_store_b32 v39, v167, s[32:33]
	.Lbf16w32_stored1:
	s_waitcnt_vscnt null, 0x0
	.Lbf16w32_end:
	s_endpgm
.Lgemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151_end:
.size gemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151, .Lgemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151_end-gemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151
	.amdhsa_group_segment_fixed_size 8192
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
	.amdhsa_user_sgpr_count 2
	.amdhsa_user_sgpr_dispatch_ptr 0
	.amdhsa_user_sgpr_queue_ptr 0
	.amdhsa_user_sgpr_kernarg_segment_ptr 1
	.amdhsa_user_sgpr_dispatch_id 0
	.amdhsa_user_sgpr_private_segment_size 0
	.amdhsa_wavefront_size32 1
	.amdhsa_uses_dynamic_stack 0
	.amdhsa_enable_private_segment 0
	.amdhsa_system_sgpr_workgroup_id_x 1
	.amdhsa_system_sgpr_workgroup_id_y 0
	.amdhsa_system_sgpr_workgroup_id_z 0
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 168
	.amdhsa_next_free_sgpr 64
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151_end-gemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.text
.protected gemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151
.globl gemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151
.p2align 8
.type gemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151,@function
gemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b128 s[16:19], s[0:1], 0x20
	v_lshrrev_b32_e32 v13, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s21, v13
	v_and_b32_e32 v5, 31, v0
	s_mov_b32 s20, s2
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s44, s20, 1
	s_cmp_ge_u32 s44, s14
	s_cbranch_scc1 .Lbf16hc16_end
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lbf16hc16_end
	s_cmp_gt_u32 s18, 16
	s_cbranch_scc1 .Lbf16hc16_end
	s_and_b32 s28, s15, 31
	s_cmp_lg_u32 s28, 0
	s_cbranch_scc1 .Lbf16hc16_end
	s_lshr_b32 s22, s21, 2
	s_and_b32 s23, s21, 3
	s_add_i32 s24, s44, s22
	s_cmp_lt_u32 s24, s14
	s_cselect_b32 s25, 1, 0
	s_lshr_b32 s26, s15, 2
	s_add_i32 s27, s26, 0xff
	s_lshr_b32 s27, s27, 8
	s_lshl_b32 s48, s26, 2
	s_lshl_b32 s41, s15, 1
	s_mul_i32 s30, s24, s41
	s_mul_hi_u32 s31, s24, s41
	s_mov_b64 s[4:5], s[8:9]
	s_add_u32 s4, s4, s30
	s_addc_u32 s5, s5, s31
	s_mul_i32 s47, s23, s26
	s_lshl_b32 s47, s47, 1
	s_add_u32 s4, s4, s47
	s_addc_u32 s5, s5, 0
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_lshl_b32 s37, s15, 2
	s_lshl_b32 s38, s14, 2
	s_lshl_b32 s39, s37, 1
	s_mul_i32 s28, s22, s37
	s_mul_i32 s29, s23, s48
	s_add_i32 s40, s28, s29
	s_mul_i32 s42, s22, 0x100
	s_lshl_b32 s28, s23, 2
	s_add_i32 s42, s42, s28
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v12, 2, v5
	s_lshl_b32 s28, s21, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v9, s28, v12
	s_lshl_b32 s28, s23, 10
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v10, s28, v12
	v_mov_b32_e32 v15, s42
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	s_mov_b32 s36, 0
	s_cmp_eq_u32 s27, 0
	s_cbranch_scc1 .Lbf16hc16_fold
	.Lbf16hc16_tile:
	s_lshl_b32 s34, s36, 9
	s_lshl_b32 s35, s36, 10
	s_lshl_b32 s33, s36, 8
	s_sub_i32 s33, s26, s33
	s_lshr_b32 s32, s33, 3
	s_lshl_b32 s32, -1, s32
	s_not_b32 s32, s32
	s_cmp_ge_u32 s33, 0x100
	s_cselect_b32 s32, -1, s32
	s_add_i32 s28, s35, s40
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s28, v3
	v_lshlrev_b32_e32 v1, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s34, v1
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc16_wdone
	s_mov_b32 exec_lo, s32
	buffer_load_b128 v[16:19], v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_wdone:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 0
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc16_st0_h0_nofill
	s_mul_i32 s29, s37, 0
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_st0_h0_nofill:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 2
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc16_st0_h1_nofill
	s_mul_i32 s29, s37, 0
	s_add_i32 s29, s29, s39
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24 offset:8192
	ds_store_b32 v9, v25 offset:8320
	ds_store_b32 v9, v26 offset:8448
	ds_store_b32 v9, v27 offset:8576
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:8704
	ds_store_b32 v9, v29 offset:8832
	ds_store_b32 v9, v30 offset:8960
	ds_store_b32 v9, v31 offset:9088
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_st0_h1_nofill:
	s_barrier
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc16_st0_cdone
	s_mov_b32 exec_lo, s32
	ds_load_b32 v24, v10
	ds_load_b32 v25, v10 offset:128
	ds_load_b32 v26, v10 offset:256
	ds_load_b32 v27, v10 offset:384
	ds_load_b32 v28, v10 offset:512
	ds_load_b32 v29, v10 offset:640
	ds_load_b32 v30, v10 offset:768
	ds_load_b32 v31, v10 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s18, 1
	s_cbranch_scc1 .Lbf16hc16_st0_cdone
	ds_load_b32 v24, v10 offset:4096
	ds_load_b32 v25, v10 offset:4224
	ds_load_b32 v26, v10 offset:4352
	ds_load_b32 v27, v10 offset:4480
	ds_load_b32 v28, v10 offset:4608
	ds_load_b32 v29, v10 offset:4736
	ds_load_b32 v30, v10 offset:4864
	ds_load_b32 v31, v10 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	s_cmp_le_u32 s18, 2
	s_cbranch_scc1 .Lbf16hc16_st0_cdone
	ds_load_b32 v24, v10 offset:8192
	ds_load_b32 v25, v10 offset:8320
	ds_load_b32 v26, v10 offset:8448
	ds_load_b32 v27, v10 offset:8576
	ds_load_b32 v28, v10 offset:8704
	ds_load_b32 v29, v10 offset:8832
	ds_load_b32 v30, v10 offset:8960
	ds_load_b32 v31, v10 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v48, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v49, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v50, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v51, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_cmp_le_u32 s18, 3
	s_cbranch_scc1 .Lbf16hc16_st0_cdone
	ds_load_b32 v24, v10 offset:12288
	ds_load_b32 v25, v10 offset:12416
	ds_load_b32 v26, v10 offset:12544
	ds_load_b32 v27, v10 offset:12672
	ds_load_b32 v28, v10 offset:12800
	ds_load_b32 v29, v10 offset:12928
	ds_load_b32 v30, v10 offset:13056
	ds_load_b32 v31, v10 offset:13184
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v52, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v53, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v54, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v55, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	.Lbf16hc16_st0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16hc16_st0_done:
	s_cmp_le_u32 s18, 4
	s_cbranch_scc1 .Lbf16hc16_st1_done
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 4
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc16_st1_h0_nofill
	s_mul_i32 s29, s37, 4
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_st1_h0_nofill:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 6
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc16_st1_h1_nofill
	s_mul_i32 s29, s37, 4
	s_add_i32 s29, s29, s39
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24 offset:8192
	ds_store_b32 v9, v25 offset:8320
	ds_store_b32 v9, v26 offset:8448
	ds_store_b32 v9, v27 offset:8576
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:8704
	ds_store_b32 v9, v29 offset:8832
	ds_store_b32 v9, v30 offset:8960
	ds_store_b32 v9, v31 offset:9088
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_st1_h1_nofill:
	s_barrier
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc16_st1_cdone
	s_mov_b32 exec_lo, s32
	s_cmp_le_u32 s18, 4
	s_cbranch_scc1 .Lbf16hc16_st1_cdone
	ds_load_b32 v24, v10
	ds_load_b32 v25, v10 offset:128
	ds_load_b32 v26, v10 offset:256
	ds_load_b32 v27, v10 offset:384
	ds_load_b32 v28, v10 offset:512
	ds_load_b32 v29, v10 offset:640
	ds_load_b32 v30, v10 offset:768
	ds_load_b32 v31, v10 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v56, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v57, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v58, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v59, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s18, 5
	s_cbranch_scc1 .Lbf16hc16_st1_cdone
	ds_load_b32 v24, v10 offset:4096
	ds_load_b32 v25, v10 offset:4224
	ds_load_b32 v26, v10 offset:4352
	ds_load_b32 v27, v10 offset:4480
	ds_load_b32 v28, v10 offset:4608
	ds_load_b32 v29, v10 offset:4736
	ds_load_b32 v30, v10 offset:4864
	ds_load_b32 v31, v10 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v60, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v61, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v62, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v63, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	s_cmp_le_u32 s18, 6
	s_cbranch_scc1 .Lbf16hc16_st1_cdone
	ds_load_b32 v24, v10 offset:8192
	ds_load_b32 v25, v10 offset:8320
	ds_load_b32 v26, v10 offset:8448
	ds_load_b32 v27, v10 offset:8576
	ds_load_b32 v28, v10 offset:8704
	ds_load_b32 v29, v10 offset:8832
	ds_load_b32 v30, v10 offset:8960
	ds_load_b32 v31, v10 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v64, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v65, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v66, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v67, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v31, v23
	s_cmp_le_u32 s18, 7
	s_cbranch_scc1 .Lbf16hc16_st1_cdone
	ds_load_b32 v24, v10 offset:12288
	ds_load_b32 v25, v10 offset:12416
	ds_load_b32 v26, v10 offset:12544
	ds_load_b32 v27, v10 offset:12672
	ds_load_b32 v28, v10 offset:12800
	ds_load_b32 v29, v10 offset:12928
	ds_load_b32 v30, v10 offset:13056
	ds_load_b32 v31, v10 offset:13184
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v68, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v69, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v70, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v71, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v31, v23
	.Lbf16hc16_st1_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16hc16_st1_done:
	s_cmp_le_u32 s18, 8
	s_cbranch_scc1 .Lbf16hc16_st2_done
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 8
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc16_st2_h0_nofill
	s_mul_i32 s29, s37, 8
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_st2_h0_nofill:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 10
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc16_st2_h1_nofill
	s_mul_i32 s29, s37, 8
	s_add_i32 s29, s29, s39
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24 offset:8192
	ds_store_b32 v9, v25 offset:8320
	ds_store_b32 v9, v26 offset:8448
	ds_store_b32 v9, v27 offset:8576
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:8704
	ds_store_b32 v9, v29 offset:8832
	ds_store_b32 v9, v30 offset:8960
	ds_store_b32 v9, v31 offset:9088
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_st2_h1_nofill:
	s_barrier
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc16_st2_cdone
	s_mov_b32 exec_lo, s32
	s_cmp_le_u32 s18, 8
	s_cbranch_scc1 .Lbf16hc16_st2_cdone
	ds_load_b32 v24, v10
	ds_load_b32 v25, v10 offset:128
	ds_load_b32 v26, v10 offset:256
	ds_load_b32 v27, v10 offset:384
	ds_load_b32 v28, v10 offset:512
	ds_load_b32 v29, v10 offset:640
	ds_load_b32 v30, v10 offset:768
	ds_load_b32 v31, v10 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v72, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v73, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v74, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v75, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v72, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v73, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v74, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v75, v31, v23
	s_cmp_le_u32 s18, 9
	s_cbranch_scc1 .Lbf16hc16_st2_cdone
	ds_load_b32 v24, v10 offset:4096
	ds_load_b32 v25, v10 offset:4224
	ds_load_b32 v26, v10 offset:4352
	ds_load_b32 v27, v10 offset:4480
	ds_load_b32 v28, v10 offset:4608
	ds_load_b32 v29, v10 offset:4736
	ds_load_b32 v30, v10 offset:4864
	ds_load_b32 v31, v10 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v76, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v77, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v78, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v79, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v76, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v77, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v78, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v79, v31, v23
	s_cmp_le_u32 s18, 10
	s_cbranch_scc1 .Lbf16hc16_st2_cdone
	ds_load_b32 v24, v10 offset:8192
	ds_load_b32 v25, v10 offset:8320
	ds_load_b32 v26, v10 offset:8448
	ds_load_b32 v27, v10 offset:8576
	ds_load_b32 v28, v10 offset:8704
	ds_load_b32 v29, v10 offset:8832
	ds_load_b32 v30, v10 offset:8960
	ds_load_b32 v31, v10 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v80, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v81, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v82, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v83, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v80, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v81, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v82, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v83, v31, v23
	s_cmp_le_u32 s18, 11
	s_cbranch_scc1 .Lbf16hc16_st2_cdone
	ds_load_b32 v24, v10 offset:12288
	ds_load_b32 v25, v10 offset:12416
	ds_load_b32 v26, v10 offset:12544
	ds_load_b32 v27, v10 offset:12672
	ds_load_b32 v28, v10 offset:12800
	ds_load_b32 v29, v10 offset:12928
	ds_load_b32 v30, v10 offset:13056
	ds_load_b32 v31, v10 offset:13184
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v84, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v85, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v86, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v87, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v84, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v85, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v86, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v87, v31, v23
	.Lbf16hc16_st2_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16hc16_st2_done:
	s_cmp_le_u32 s18, 12
	s_cbranch_scc1 .Lbf16hc16_st3_done
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 12
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc16_st3_h0_nofill
	s_mul_i32 s29, s37, 12
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_st3_h0_nofill:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 14
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc16_st3_h1_nofill
	s_mul_i32 s29, s37, 12
	s_add_i32 s29, s29, s39
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24 offset:8192
	ds_store_b32 v9, v25 offset:8320
	ds_store_b32 v9, v26 offset:8448
	ds_store_b32 v9, v27 offset:8576
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:8704
	ds_store_b32 v9, v29 offset:8832
	ds_store_b32 v9, v30 offset:8960
	ds_store_b32 v9, v31 offset:9088
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_st3_h1_nofill:
	s_barrier
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc16_st3_cdone
	s_mov_b32 exec_lo, s32
	s_cmp_le_u32 s18, 12
	s_cbranch_scc1 .Lbf16hc16_st3_cdone
	ds_load_b32 v24, v10
	ds_load_b32 v25, v10 offset:128
	ds_load_b32 v26, v10 offset:256
	ds_load_b32 v27, v10 offset:384
	ds_load_b32 v28, v10 offset:512
	ds_load_b32 v29, v10 offset:640
	ds_load_b32 v30, v10 offset:768
	ds_load_b32 v31, v10 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v88, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v89, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v90, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v91, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s18, 13
	s_cbranch_scc1 .Lbf16hc16_st3_cdone
	ds_load_b32 v24, v10 offset:4096
	ds_load_b32 v25, v10 offset:4224
	ds_load_b32 v26, v10 offset:4352
	ds_load_b32 v27, v10 offset:4480
	ds_load_b32 v28, v10 offset:4608
	ds_load_b32 v29, v10 offset:4736
	ds_load_b32 v30, v10 offset:4864
	ds_load_b32 v31, v10 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v92, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v93, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v94, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v95, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v31, v23
	s_cmp_le_u32 s18, 14
	s_cbranch_scc1 .Lbf16hc16_st3_cdone
	ds_load_b32 v24, v10 offset:8192
	ds_load_b32 v25, v10 offset:8320
	ds_load_b32 v26, v10 offset:8448
	ds_load_b32 v27, v10 offset:8576
	ds_load_b32 v28, v10 offset:8704
	ds_load_b32 v29, v10 offset:8832
	ds_load_b32 v30, v10 offset:8960
	ds_load_b32 v31, v10 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v96, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v97, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v98, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v99, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v31, v23
	s_cmp_le_u32 s18, 15
	s_cbranch_scc1 .Lbf16hc16_st3_cdone
	ds_load_b32 v24, v10 offset:12288
	ds_load_b32 v25, v10 offset:12416
	ds_load_b32 v26, v10 offset:12544
	ds_load_b32 v27, v10 offset:12672
	ds_load_b32 v28, v10 offset:12800
	ds_load_b32 v29, v10 offset:12928
	ds_load_b32 v30, v10 offset:13056
	ds_load_b32 v31, v10 offset:13184
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v100, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v101, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v102, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v103, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v31, v23
	.Lbf16hc16_st3_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16hc16_st3_done:
	s_add_i32 s36, s36, 1
	s_cmp_lt_u32 s36, s27
	s_cbranch_scc1 .Lbf16hc16_tile
	.Lbf16hc16_fold:
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc16_pdone
	v_add_f32_e32 v41, v41, v40
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v41
	v_add_f32_e32 v45, v45, v44
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v45
	v_add_f32_e32 v49, v49, v48
	v_add_f32_e32 v51, v51, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v49
	v_add_f32_e32 v53, v53, v52
	v_add_f32_e32 v55, v55, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v53
	v_add_f32_e32 v57, v57, v56
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v57
	v_add_f32_e32 v61, v61, v60
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v61
	v_add_f32_e32 v65, v65, v64
	v_add_f32_e32 v67, v67, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v65
	v_add_f32_e32 v69, v69, v68
	v_add_f32_e32 v71, v71, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v69
	v_add_f32_e32 v73, v73, v72
	v_add_f32_e32 v75, v75, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v75, v75, v73
	v_add_f32_e32 v77, v77, v76
	v_add_f32_e32 v79, v79, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v79, v79, v77
	v_add_f32_e32 v81, v81, v80
	v_add_f32_e32 v83, v83, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v83, v83, v81
	v_add_f32_e32 v85, v85, v84
	v_add_f32_e32 v87, v87, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v87, v87, v85
	v_add_f32_e32 v89, v89, v88
	v_add_f32_e32 v91, v91, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v91, v91, v89
	v_add_f32_e32 v93, v93, v92
	v_add_f32_e32 v95, v95, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v95, v95, v93
	v_add_f32_e32 v97, v97, v96
	v_add_f32_e32 v99, v99, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v99, v99, v97
	v_add_f32_e32 v101, v101, v100
	v_add_f32_e32 v103, v103, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v103, v103, v101
	ds_swizzle_b32 v24, v43 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v47 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v51 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v55 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v59 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v63 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v71 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v75 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v79 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v83 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v87 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v36, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v37, v95 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v38, v99 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v39, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmpx_eq_u32_e32 0, v5
	ds_store_b32 v15, v43
	s_cmp_le_u32 s18, 1
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v47 offset:16
	s_cmp_le_u32 s18, 2
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v51 offset:32
	s_cmp_le_u32 s18, 3
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v55 offset:48
	s_cmp_le_u32 s18, 4
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v59 offset:64
	s_cmp_le_u32 s18, 5
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v63 offset:80
	s_cmp_le_u32 s18, 6
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v67 offset:96
	s_cmp_le_u32 s18, 7
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v71 offset:112
	s_cmp_le_u32 s18, 8
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v75 offset:128
	s_cmp_le_u32 s18, 9
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v79 offset:144
	s_cmp_le_u32 s18, 10
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v83 offset:160
	s_cmp_le_u32 s18, 11
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v87 offset:176
	s_cmp_le_u32 s18, 12
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v91 offset:192
	s_cmp_le_u32 s18, 13
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v95 offset:208
	s_cmp_le_u32 s18, 14
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v99 offset:224
	s_cmp_le_u32 s18, 15
	s_cbranch_scc1 .Lbf16hc16_pstored
	ds_store_b32 v15, v103 offset:240
	.Lbf16hc16_pstored:
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_pdone:
	s_barrier
	s_add_i32 s45, s21, 0
	s_cmp_ge_u32 s45, s18
	s_cbranch_scc1 .Lbf16hc16_iskip0
	s_lshl_b32 s28, s45, 4
	v_mov_b32_e32 v40, s28
	ds_load_b128 v[41:44], v40
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s64, v41
	v_readfirstlane_b32 s65, v42
	v_readfirstlane_b32 s66, v43
	v_readfirstlane_b32 s67, v44
	s_add_f32 s69, s64, s65
	s_add_f32 s70, s66, s67
	s_delay_alu instid0(SALU_CYCLE_3)
	s_add_f32 s68, s69, s70
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lbf16hc16_item0_done
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s68, s16, s68
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s69, s68, 0xbfb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s70, s69, 0x80000000
	s_rndne_f32 s71, s69
	s_fmamk_f32 s70, s68, 0xbfb8aa3b, s70
	s_cmp_ngt_f32 s68, 0x42ce8ed0
	s_sub_f32 s69, s69, s71
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s70, s68, 0xb2a5705f, s70
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_nlt_f32 s68, 0xc2b17218
	s_add_f32 s69, s69, s70
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v24, s69
	s_cvt_i32_f32 s69, s71
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, s69
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s68, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s68, v24
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	.Lbf16hc16_item0_done:
	s_mul_i32 s46, s45, s38
	s_add_i32 s28, s44, 0
	s_lshl_b32 s28, s28, 2
	s_add_i32 s46, s46, s28
	v_mov_b32_e32 v45, s46
	v_mov_b32_e32 v46, s68
	s_mov_b32 exec_lo, 1
	global_store_b32 v45, v46, s[12:13]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_iskip0:
	s_add_i32 s45, s21, 8
	s_cmp_ge_u32 s45, s18
	s_cbranch_scc1 .Lbf16hc16_iskip1
	s_lshl_b32 s28, s45, 4
	v_mov_b32_e32 v40, s28
	ds_load_b128 v[41:44], v40
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s64, v41
	v_readfirstlane_b32 s65, v42
	v_readfirstlane_b32 s66, v43
	v_readfirstlane_b32 s67, v44
	s_add_f32 s69, s64, s65
	s_add_f32 s70, s66, s67
	s_delay_alu instid0(SALU_CYCLE_3)
	s_add_f32 s68, s69, s70
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lbf16hc16_item1_done
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s68, s16, s68
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s69, s68, 0xbfb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s70, s69, 0x80000000
	s_rndne_f32 s71, s69
	s_fmamk_f32 s70, s68, 0xbfb8aa3b, s70
	s_cmp_ngt_f32 s68, 0x42ce8ed0
	s_sub_f32 s69, s69, s71
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s70, s68, 0xb2a5705f, s70
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_nlt_f32 s68, 0xc2b17218
	s_add_f32 s69, s69, s70
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v24, s69
	s_cvt_i32_f32 s69, s71
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, s69
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s68, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s68, v24
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	.Lbf16hc16_item1_done:
	s_mul_i32 s46, s45, s38
	s_add_i32 s28, s44, 0
	s_lshl_b32 s28, s28, 2
	s_add_i32 s46, s46, s28
	v_mov_b32_e32 v45, s46
	v_mov_b32_e32 v46, s68
	s_mov_b32 exec_lo, 1
	global_store_b32 v45, v46, s[12:13]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_iskip1:
	s_add_i32 s45, s21, 0
	s_cmp_ge_u32 s45, s18
	s_cbranch_scc1 .Lbf16hc16_iskip2
	s_add_i32 s28, s44, 1
	s_cmp_ge_u32 s28, s14
	s_cbranch_scc1 .Lbf16hc16_iskip2
	s_lshl_b32 s28, s45, 4
	s_add_i32 s28, s28, 0x100
	v_mov_b32_e32 v40, s28
	ds_load_b128 v[41:44], v40
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s64, v41
	v_readfirstlane_b32 s65, v42
	v_readfirstlane_b32 s66, v43
	v_readfirstlane_b32 s67, v44
	s_add_f32 s69, s64, s65
	s_add_f32 s70, s66, s67
	s_delay_alu instid0(SALU_CYCLE_3)
	s_add_f32 s68, s69, s70
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lbf16hc16_item2_done
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s68, s16, s68
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s69, s68, 0xbfb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s70, s69, 0x80000000
	s_rndne_f32 s71, s69
	s_fmamk_f32 s70, s68, 0xbfb8aa3b, s70
	s_cmp_ngt_f32 s68, 0x42ce8ed0
	s_sub_f32 s69, s69, s71
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s70, s68, 0xb2a5705f, s70
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_nlt_f32 s68, 0xc2b17218
	s_add_f32 s69, s69, s70
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v24, s69
	s_cvt_i32_f32 s69, s71
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, s69
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s68, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s68, v24
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	.Lbf16hc16_item2_done:
	s_mul_i32 s46, s45, s38
	s_add_i32 s28, s44, 1
	s_lshl_b32 s28, s28, 2
	s_add_i32 s46, s46, s28
	v_mov_b32_e32 v45, s46
	v_mov_b32_e32 v46, s68
	s_mov_b32 exec_lo, 1
	global_store_b32 v45, v46, s[12:13]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_iskip2:
	s_add_i32 s45, s21, 8
	s_cmp_ge_u32 s45, s18
	s_cbranch_scc1 .Lbf16hc16_iskip3
	s_add_i32 s28, s44, 1
	s_cmp_ge_u32 s28, s14
	s_cbranch_scc1 .Lbf16hc16_iskip3
	s_lshl_b32 s28, s45, 4
	s_add_i32 s28, s28, 0x100
	v_mov_b32_e32 v40, s28
	ds_load_b128 v[41:44], v40
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s64, v41
	v_readfirstlane_b32 s65, v42
	v_readfirstlane_b32 s66, v43
	v_readfirstlane_b32 s67, v44
	s_add_f32 s69, s64, s65
	s_add_f32 s70, s66, s67
	s_delay_alu instid0(SALU_CYCLE_3)
	s_add_f32 s68, s69, s70
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lbf16hc16_item3_done
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s68, s16, s68
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s69, s68, 0xbfb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s70, s69, 0x80000000
	s_rndne_f32 s71, s69
	s_fmamk_f32 s70, s68, 0xbfb8aa3b, s70
	s_cmp_ngt_f32 s68, 0x42ce8ed0
	s_sub_f32 s69, s69, s71
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s70, s68, 0xb2a5705f, s70
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_nlt_f32 s68, 0xc2b17218
	s_add_f32 s69, s69, s70
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v24, s69
	s_cvt_i32_f32 s69, s71
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, s69
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s68, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s68, v24
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	.Lbf16hc16_item3_done:
	s_mul_i32 s46, s45, s38
	s_add_i32 s28, s44, 1
	s_lshl_b32 s28, s28, 2
	s_add_i32 s46, s46, s28
	v_mov_b32_e32 v45, s46
	v_mov_b32_e32 v46, s68
	s_mov_b32 exec_lo, 1
	global_store_b32 v45, v46, s[12:13]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16hc16_iskip3:
	s_barrier
	.Lbf16hc16_end:
	s_endpgm
.Lgemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151_end:
.size gemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151, .Lgemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151_end-gemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151
	.amdhsa_group_segment_fixed_size 16384
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 48
	.amdhsa_user_sgpr_count 2
	.amdhsa_user_sgpr_dispatch_ptr 0
	.amdhsa_user_sgpr_queue_ptr 0
	.amdhsa_user_sgpr_kernarg_segment_ptr 1
	.amdhsa_user_sgpr_dispatch_id 0
	.amdhsa_user_sgpr_private_segment_size 0
	.amdhsa_wavefront_size32 1
	.amdhsa_uses_dynamic_stack 0
	.amdhsa_enable_private_segment 0
	.amdhsa_system_sgpr_workgroup_id_x 1
	.amdhsa_system_sgpr_workgroup_id_y 0
	.amdhsa_system_sgpr_workgroup_id_z 0
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 104
	.amdhsa_next_free_sgpr 80
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151_end-gemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.text
.protected gemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151
.globl gemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151
.p2align 8
.type gemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151,@function
gemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b128 s[16:19], s[0:1], 0x20
	v_lshrrev_b32_e32 v13, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s21, v13
	v_and_b32_e32 v5, 31, v0
	s_mov_b32 s20, s2
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s44, s20, 1
	s_cmp_ge_u32 s44, s14
	s_cbranch_scc1 .Lbf16hc32_end
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lbf16hc32_end
	s_cmp_gt_u32 s18, 32
	s_cbranch_scc1 .Lbf16hc32_end
	s_and_b32 s28, s15, 31
	s_cmp_lg_u32 s28, 0
	s_cbranch_scc1 .Lbf16hc32_end
	s_lshr_b32 s22, s21, 2
	s_and_b32 s23, s21, 3
	s_add_i32 s24, s44, s22
	s_cmp_lt_u32 s24, s14
	s_cselect_b32 s25, 1, 0
	s_lshr_b32 s26, s15, 2
	s_add_i32 s27, s26, 0xff
	s_lshr_b32 s27, s27, 8
	s_lshl_b32 s48, s26, 2
	s_lshl_b32 s41, s15, 1
	s_mul_i32 s30, s24, s41
	s_mul_hi_u32 s31, s24, s41
	s_mov_b64 s[4:5], s[8:9]
	s_add_u32 s4, s4, s30
	s_addc_u32 s5, s5, s31
	s_mul_i32 s47, s23, s26
	s_lshl_b32 s47, s47, 1
	s_add_u32 s4, s4, s47
	s_addc_u32 s5, s5, 0
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_lshl_b32 s37, s15, 2
	s_lshl_b32 s38, s14, 2
	s_lshl_b32 s39, s37, 1
	s_mul_i32 s28, s22, s37
	s_mul_i32 s29, s23, s48
	s_add_i32 s40, s28, s29
	s_mul_i32 s42, s22, 0x200
	s_lshl_b32 s28, s23, 2
	s_add_i32 s42, s42, s28
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v12, 2, v5
	s_lshl_b32 s28, s21, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v9, s28, v12
	s_lshl_b32 s28, s23, 10
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v10, s28, v12
	v_mov_b32_e32 v15, s42
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	s_mov_b32 s36, 0
	s_cmp_eq_u32 s27, 0
	s_cbranch_scc1 .Lbf16hc32_fold
	.Lbf16hc32_tile:
	s_lshl_b32 s34, s36, 9
	s_lshl_b32 s35, s36, 10
	s_lshl_b32 s33, s36, 8
	s_sub_i32 s33, s26, s33
	s_lshr_b32 s32, s33, 3
	s_lshl_b32 s32, -1, s32
	s_not_b32 s32, s32
	s_cmp_ge_u32 s33, 0x100
	s_cselect_b32 s32, -1, s32
	s_add_i32 s28, s35, s40
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s28, v3
	v_lshlrev_b32_e32 v1, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s34, v1
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc32_wdone
	s_mov_b32 exec_lo, s32
	buffer_load_b128 v[16:19], v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_wdone:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 0
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st0_h0_nofill
	s_mul_i32 s29, s37, 0
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st0_h0_nofill:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 2
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st0_h1_nofill
	s_mul_i32 s29, s37, 0
	s_add_i32 s29, s29, s39
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24 offset:8192
	ds_store_b32 v9, v25 offset:8320
	ds_store_b32 v9, v26 offset:8448
	ds_store_b32 v9, v27 offset:8576
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:8704
	ds_store_b32 v9, v29 offset:8832
	ds_store_b32 v9, v30 offset:8960
	ds_store_b32 v9, v31 offset:9088
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st0_h1_nofill:
	s_barrier
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc32_st0_cdone
	s_mov_b32 exec_lo, s32
	ds_load_b32 v24, v10
	ds_load_b32 v25, v10 offset:128
	ds_load_b32 v26, v10 offset:256
	ds_load_b32 v27, v10 offset:384
	ds_load_b32 v28, v10 offset:512
	ds_load_b32 v29, v10 offset:640
	ds_load_b32 v30, v10 offset:768
	ds_load_b32 v31, v10 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v40, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v41, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v42, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v43, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s18, 1
	s_cbranch_scc1 .Lbf16hc32_st0_cdone
	ds_load_b32 v24, v10 offset:4096
	ds_load_b32 v25, v10 offset:4224
	ds_load_b32 v26, v10 offset:4352
	ds_load_b32 v27, v10 offset:4480
	ds_load_b32 v28, v10 offset:4608
	ds_load_b32 v29, v10 offset:4736
	ds_load_b32 v30, v10 offset:4864
	ds_load_b32 v31, v10 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v44, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v45, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v46, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v47, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	s_cmp_le_u32 s18, 2
	s_cbranch_scc1 .Lbf16hc32_st0_cdone
	ds_load_b32 v24, v10 offset:8192
	ds_load_b32 v25, v10 offset:8320
	ds_load_b32 v26, v10 offset:8448
	ds_load_b32 v27, v10 offset:8576
	ds_load_b32 v28, v10 offset:8704
	ds_load_b32 v29, v10 offset:8832
	ds_load_b32 v30, v10 offset:8960
	ds_load_b32 v31, v10 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v48, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v49, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v50, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v51, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_cmp_le_u32 s18, 3
	s_cbranch_scc1 .Lbf16hc32_st0_cdone
	ds_load_b32 v24, v10 offset:12288
	ds_load_b32 v25, v10 offset:12416
	ds_load_b32 v26, v10 offset:12544
	ds_load_b32 v27, v10 offset:12672
	ds_load_b32 v28, v10 offset:12800
	ds_load_b32 v29, v10 offset:12928
	ds_load_b32 v30, v10 offset:13056
	ds_load_b32 v31, v10 offset:13184
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v52, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v53, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v54, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v55, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	.Lbf16hc32_st0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16hc32_st0_done:
	s_cmp_le_u32 s18, 4
	s_cbranch_scc1 .Lbf16hc32_st1_done
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 4
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st1_h0_nofill
	s_mul_i32 s29, s37, 4
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st1_h0_nofill:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 6
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st1_h1_nofill
	s_mul_i32 s29, s37, 4
	s_add_i32 s29, s29, s39
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24 offset:8192
	ds_store_b32 v9, v25 offset:8320
	ds_store_b32 v9, v26 offset:8448
	ds_store_b32 v9, v27 offset:8576
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:8704
	ds_store_b32 v9, v29 offset:8832
	ds_store_b32 v9, v30 offset:8960
	ds_store_b32 v9, v31 offset:9088
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st1_h1_nofill:
	s_barrier
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc32_st1_cdone
	s_mov_b32 exec_lo, s32
	s_cmp_le_u32 s18, 4
	s_cbranch_scc1 .Lbf16hc32_st1_cdone
	ds_load_b32 v24, v10
	ds_load_b32 v25, v10 offset:128
	ds_load_b32 v26, v10 offset:256
	ds_load_b32 v27, v10 offset:384
	ds_load_b32 v28, v10 offset:512
	ds_load_b32 v29, v10 offset:640
	ds_load_b32 v30, v10 offset:768
	ds_load_b32 v31, v10 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v56, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v57, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v58, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v59, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s18, 5
	s_cbranch_scc1 .Lbf16hc32_st1_cdone
	ds_load_b32 v24, v10 offset:4096
	ds_load_b32 v25, v10 offset:4224
	ds_load_b32 v26, v10 offset:4352
	ds_load_b32 v27, v10 offset:4480
	ds_load_b32 v28, v10 offset:4608
	ds_load_b32 v29, v10 offset:4736
	ds_load_b32 v30, v10 offset:4864
	ds_load_b32 v31, v10 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v60, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v61, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v62, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v63, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	s_cmp_le_u32 s18, 6
	s_cbranch_scc1 .Lbf16hc32_st1_cdone
	ds_load_b32 v24, v10 offset:8192
	ds_load_b32 v25, v10 offset:8320
	ds_load_b32 v26, v10 offset:8448
	ds_load_b32 v27, v10 offset:8576
	ds_load_b32 v28, v10 offset:8704
	ds_load_b32 v29, v10 offset:8832
	ds_load_b32 v30, v10 offset:8960
	ds_load_b32 v31, v10 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v64, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v65, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v66, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v67, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v31, v23
	s_cmp_le_u32 s18, 7
	s_cbranch_scc1 .Lbf16hc32_st1_cdone
	ds_load_b32 v24, v10 offset:12288
	ds_load_b32 v25, v10 offset:12416
	ds_load_b32 v26, v10 offset:12544
	ds_load_b32 v27, v10 offset:12672
	ds_load_b32 v28, v10 offset:12800
	ds_load_b32 v29, v10 offset:12928
	ds_load_b32 v30, v10 offset:13056
	ds_load_b32 v31, v10 offset:13184
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v68, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v69, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v70, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v71, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v31, v23
	.Lbf16hc32_st1_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16hc32_st1_done:
	s_cmp_le_u32 s18, 8
	s_cbranch_scc1 .Lbf16hc32_st2_done
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 8
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st2_h0_nofill
	s_mul_i32 s29, s37, 8
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st2_h0_nofill:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 10
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st2_h1_nofill
	s_mul_i32 s29, s37, 8
	s_add_i32 s29, s29, s39
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24 offset:8192
	ds_store_b32 v9, v25 offset:8320
	ds_store_b32 v9, v26 offset:8448
	ds_store_b32 v9, v27 offset:8576
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:8704
	ds_store_b32 v9, v29 offset:8832
	ds_store_b32 v9, v30 offset:8960
	ds_store_b32 v9, v31 offset:9088
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st2_h1_nofill:
	s_barrier
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc32_st2_cdone
	s_mov_b32 exec_lo, s32
	s_cmp_le_u32 s18, 8
	s_cbranch_scc1 .Lbf16hc32_st2_cdone
	ds_load_b32 v24, v10
	ds_load_b32 v25, v10 offset:128
	ds_load_b32 v26, v10 offset:256
	ds_load_b32 v27, v10 offset:384
	ds_load_b32 v28, v10 offset:512
	ds_load_b32 v29, v10 offset:640
	ds_load_b32 v30, v10 offset:768
	ds_load_b32 v31, v10 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v72, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v73, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v74, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v75, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v72, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v73, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v74, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v75, v31, v23
	s_cmp_le_u32 s18, 9
	s_cbranch_scc1 .Lbf16hc32_st2_cdone
	ds_load_b32 v24, v10 offset:4096
	ds_load_b32 v25, v10 offset:4224
	ds_load_b32 v26, v10 offset:4352
	ds_load_b32 v27, v10 offset:4480
	ds_load_b32 v28, v10 offset:4608
	ds_load_b32 v29, v10 offset:4736
	ds_load_b32 v30, v10 offset:4864
	ds_load_b32 v31, v10 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v76, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v77, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v78, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v79, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v76, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v77, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v78, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v79, v31, v23
	s_cmp_le_u32 s18, 10
	s_cbranch_scc1 .Lbf16hc32_st2_cdone
	ds_load_b32 v24, v10 offset:8192
	ds_load_b32 v25, v10 offset:8320
	ds_load_b32 v26, v10 offset:8448
	ds_load_b32 v27, v10 offset:8576
	ds_load_b32 v28, v10 offset:8704
	ds_load_b32 v29, v10 offset:8832
	ds_load_b32 v30, v10 offset:8960
	ds_load_b32 v31, v10 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v80, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v81, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v82, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v83, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v80, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v81, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v82, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v83, v31, v23
	s_cmp_le_u32 s18, 11
	s_cbranch_scc1 .Lbf16hc32_st2_cdone
	ds_load_b32 v24, v10 offset:12288
	ds_load_b32 v25, v10 offset:12416
	ds_load_b32 v26, v10 offset:12544
	ds_load_b32 v27, v10 offset:12672
	ds_load_b32 v28, v10 offset:12800
	ds_load_b32 v29, v10 offset:12928
	ds_load_b32 v30, v10 offset:13056
	ds_load_b32 v31, v10 offset:13184
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v84, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v85, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v86, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v87, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v84, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v85, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v86, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v87, v31, v23
	.Lbf16hc32_st2_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16hc32_st2_done:
	s_cmp_le_u32 s18, 12
	s_cbranch_scc1 .Lbf16hc32_st3_done
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 12
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st3_h0_nofill
	s_mul_i32 s29, s37, 12
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st3_h0_nofill:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 14
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st3_h1_nofill
	s_mul_i32 s29, s37, 12
	s_add_i32 s29, s29, s39
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24 offset:8192
	ds_store_b32 v9, v25 offset:8320
	ds_store_b32 v9, v26 offset:8448
	ds_store_b32 v9, v27 offset:8576
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:8704
	ds_store_b32 v9, v29 offset:8832
	ds_store_b32 v9, v30 offset:8960
	ds_store_b32 v9, v31 offset:9088
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st3_h1_nofill:
	s_barrier
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc32_st3_cdone
	s_mov_b32 exec_lo, s32
	s_cmp_le_u32 s18, 12
	s_cbranch_scc1 .Lbf16hc32_st3_cdone
	ds_load_b32 v24, v10
	ds_load_b32 v25, v10 offset:128
	ds_load_b32 v26, v10 offset:256
	ds_load_b32 v27, v10 offset:384
	ds_load_b32 v28, v10 offset:512
	ds_load_b32 v29, v10 offset:640
	ds_load_b32 v30, v10 offset:768
	ds_load_b32 v31, v10 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v88, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v89, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v90, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v91, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s18, 13
	s_cbranch_scc1 .Lbf16hc32_st3_cdone
	ds_load_b32 v24, v10 offset:4096
	ds_load_b32 v25, v10 offset:4224
	ds_load_b32 v26, v10 offset:4352
	ds_load_b32 v27, v10 offset:4480
	ds_load_b32 v28, v10 offset:4608
	ds_load_b32 v29, v10 offset:4736
	ds_load_b32 v30, v10 offset:4864
	ds_load_b32 v31, v10 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v92, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v93, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v94, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v95, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v31, v23
	s_cmp_le_u32 s18, 14
	s_cbranch_scc1 .Lbf16hc32_st3_cdone
	ds_load_b32 v24, v10 offset:8192
	ds_load_b32 v25, v10 offset:8320
	ds_load_b32 v26, v10 offset:8448
	ds_load_b32 v27, v10 offset:8576
	ds_load_b32 v28, v10 offset:8704
	ds_load_b32 v29, v10 offset:8832
	ds_load_b32 v30, v10 offset:8960
	ds_load_b32 v31, v10 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v96, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v97, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v98, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v99, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v31, v23
	s_cmp_le_u32 s18, 15
	s_cbranch_scc1 .Lbf16hc32_st3_cdone
	ds_load_b32 v24, v10 offset:12288
	ds_load_b32 v25, v10 offset:12416
	ds_load_b32 v26, v10 offset:12544
	ds_load_b32 v27, v10 offset:12672
	ds_load_b32 v28, v10 offset:12800
	ds_load_b32 v29, v10 offset:12928
	ds_load_b32 v30, v10 offset:13056
	ds_load_b32 v31, v10 offset:13184
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v100, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v101, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v102, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v103, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v31, v23
	.Lbf16hc32_st3_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16hc32_st3_done:
	s_cmp_le_u32 s18, 16
	s_cbranch_scc1 .Lbf16hc32_st4_done
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 16
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st4_h0_nofill
	s_mul_i32 s29, s37, 16
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st4_h0_nofill:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 18
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st4_h1_nofill
	s_mul_i32 s29, s37, 16
	s_add_i32 s29, s29, s39
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24 offset:8192
	ds_store_b32 v9, v25 offset:8320
	ds_store_b32 v9, v26 offset:8448
	ds_store_b32 v9, v27 offset:8576
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:8704
	ds_store_b32 v9, v29 offset:8832
	ds_store_b32 v9, v30 offset:8960
	ds_store_b32 v9, v31 offset:9088
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st4_h1_nofill:
	s_barrier
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc32_st4_cdone
	s_mov_b32 exec_lo, s32
	s_cmp_le_u32 s18, 16
	s_cbranch_scc1 .Lbf16hc32_st4_cdone
	ds_load_b32 v24, v10
	ds_load_b32 v25, v10 offset:128
	ds_load_b32 v26, v10 offset:256
	ds_load_b32 v27, v10 offset:384
	ds_load_b32 v28, v10 offset:512
	ds_load_b32 v29, v10 offset:640
	ds_load_b32 v30, v10 offset:768
	ds_load_b32 v31, v10 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v104, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v105, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v106, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v107, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v104, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v105, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v106, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v107, v31, v23
	s_cmp_le_u32 s18, 17
	s_cbranch_scc1 .Lbf16hc32_st4_cdone
	ds_load_b32 v24, v10 offset:4096
	ds_load_b32 v25, v10 offset:4224
	ds_load_b32 v26, v10 offset:4352
	ds_load_b32 v27, v10 offset:4480
	ds_load_b32 v28, v10 offset:4608
	ds_load_b32 v29, v10 offset:4736
	ds_load_b32 v30, v10 offset:4864
	ds_load_b32 v31, v10 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v108, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v109, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v110, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v111, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v108, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v109, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v110, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v111, v31, v23
	s_cmp_le_u32 s18, 18
	s_cbranch_scc1 .Lbf16hc32_st4_cdone
	ds_load_b32 v24, v10 offset:8192
	ds_load_b32 v25, v10 offset:8320
	ds_load_b32 v26, v10 offset:8448
	ds_load_b32 v27, v10 offset:8576
	ds_load_b32 v28, v10 offset:8704
	ds_load_b32 v29, v10 offset:8832
	ds_load_b32 v30, v10 offset:8960
	ds_load_b32 v31, v10 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v112, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v113, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v114, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v115, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v112, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v113, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v114, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v115, v31, v23
	s_cmp_le_u32 s18, 19
	s_cbranch_scc1 .Lbf16hc32_st4_cdone
	ds_load_b32 v24, v10 offset:12288
	ds_load_b32 v25, v10 offset:12416
	ds_load_b32 v26, v10 offset:12544
	ds_load_b32 v27, v10 offset:12672
	ds_load_b32 v28, v10 offset:12800
	ds_load_b32 v29, v10 offset:12928
	ds_load_b32 v30, v10 offset:13056
	ds_load_b32 v31, v10 offset:13184
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v116, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v117, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v118, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v119, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v116, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v117, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v118, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v119, v31, v23
	.Lbf16hc32_st4_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16hc32_st4_done:
	s_cmp_le_u32 s18, 20
	s_cbranch_scc1 .Lbf16hc32_st5_done
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 20
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st5_h0_nofill
	s_mul_i32 s29, s37, 20
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st5_h0_nofill:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 22
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st5_h1_nofill
	s_mul_i32 s29, s37, 20
	s_add_i32 s29, s29, s39
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24 offset:8192
	ds_store_b32 v9, v25 offset:8320
	ds_store_b32 v9, v26 offset:8448
	ds_store_b32 v9, v27 offset:8576
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:8704
	ds_store_b32 v9, v29 offset:8832
	ds_store_b32 v9, v30 offset:8960
	ds_store_b32 v9, v31 offset:9088
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st5_h1_nofill:
	s_barrier
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc32_st5_cdone
	s_mov_b32 exec_lo, s32
	s_cmp_le_u32 s18, 20
	s_cbranch_scc1 .Lbf16hc32_st5_cdone
	ds_load_b32 v24, v10
	ds_load_b32 v25, v10 offset:128
	ds_load_b32 v26, v10 offset:256
	ds_load_b32 v27, v10 offset:384
	ds_load_b32 v28, v10 offset:512
	ds_load_b32 v29, v10 offset:640
	ds_load_b32 v30, v10 offset:768
	ds_load_b32 v31, v10 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v120, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v121, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v122, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v123, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v120, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v121, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v122, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v123, v31, v23
	s_cmp_le_u32 s18, 21
	s_cbranch_scc1 .Lbf16hc32_st5_cdone
	ds_load_b32 v24, v10 offset:4096
	ds_load_b32 v25, v10 offset:4224
	ds_load_b32 v26, v10 offset:4352
	ds_load_b32 v27, v10 offset:4480
	ds_load_b32 v28, v10 offset:4608
	ds_load_b32 v29, v10 offset:4736
	ds_load_b32 v30, v10 offset:4864
	ds_load_b32 v31, v10 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v124, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v125, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v126, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v127, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v124, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v125, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v126, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v127, v31, v23
	s_cmp_le_u32 s18, 22
	s_cbranch_scc1 .Lbf16hc32_st5_cdone
	ds_load_b32 v24, v10 offset:8192
	ds_load_b32 v25, v10 offset:8320
	ds_load_b32 v26, v10 offset:8448
	ds_load_b32 v27, v10 offset:8576
	ds_load_b32 v28, v10 offset:8704
	ds_load_b32 v29, v10 offset:8832
	ds_load_b32 v30, v10 offset:8960
	ds_load_b32 v31, v10 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v128, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v129, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v130, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v131, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v128, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v129, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v130, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v131, v31, v23
	s_cmp_le_u32 s18, 23
	s_cbranch_scc1 .Lbf16hc32_st5_cdone
	ds_load_b32 v24, v10 offset:12288
	ds_load_b32 v25, v10 offset:12416
	ds_load_b32 v26, v10 offset:12544
	ds_load_b32 v27, v10 offset:12672
	ds_load_b32 v28, v10 offset:12800
	ds_load_b32 v29, v10 offset:12928
	ds_load_b32 v30, v10 offset:13056
	ds_load_b32 v31, v10 offset:13184
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v132, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v133, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v134, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v135, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v132, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v133, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v134, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v135, v31, v23
	.Lbf16hc32_st5_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16hc32_st5_done:
	s_cmp_le_u32 s18, 24
	s_cbranch_scc1 .Lbf16hc32_st6_done
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 24
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st6_h0_nofill
	s_mul_i32 s29, s37, 24
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st6_h0_nofill:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 26
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st6_h1_nofill
	s_mul_i32 s29, s37, 24
	s_add_i32 s29, s29, s39
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24 offset:8192
	ds_store_b32 v9, v25 offset:8320
	ds_store_b32 v9, v26 offset:8448
	ds_store_b32 v9, v27 offset:8576
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:8704
	ds_store_b32 v9, v29 offset:8832
	ds_store_b32 v9, v30 offset:8960
	ds_store_b32 v9, v31 offset:9088
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st6_h1_nofill:
	s_barrier
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc32_st6_cdone
	s_mov_b32 exec_lo, s32
	s_cmp_le_u32 s18, 24
	s_cbranch_scc1 .Lbf16hc32_st6_cdone
	ds_load_b32 v24, v10
	ds_load_b32 v25, v10 offset:128
	ds_load_b32 v26, v10 offset:256
	ds_load_b32 v27, v10 offset:384
	ds_load_b32 v28, v10 offset:512
	ds_load_b32 v29, v10 offset:640
	ds_load_b32 v30, v10 offset:768
	ds_load_b32 v31, v10 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v136, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v137, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v138, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v139, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v136, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v137, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v138, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v139, v31, v23
	s_cmp_le_u32 s18, 25
	s_cbranch_scc1 .Lbf16hc32_st6_cdone
	ds_load_b32 v24, v10 offset:4096
	ds_load_b32 v25, v10 offset:4224
	ds_load_b32 v26, v10 offset:4352
	ds_load_b32 v27, v10 offset:4480
	ds_load_b32 v28, v10 offset:4608
	ds_load_b32 v29, v10 offset:4736
	ds_load_b32 v30, v10 offset:4864
	ds_load_b32 v31, v10 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v140, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v141, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v142, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v143, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v140, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v141, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v142, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v143, v31, v23
	s_cmp_le_u32 s18, 26
	s_cbranch_scc1 .Lbf16hc32_st6_cdone
	ds_load_b32 v24, v10 offset:8192
	ds_load_b32 v25, v10 offset:8320
	ds_load_b32 v26, v10 offset:8448
	ds_load_b32 v27, v10 offset:8576
	ds_load_b32 v28, v10 offset:8704
	ds_load_b32 v29, v10 offset:8832
	ds_load_b32 v30, v10 offset:8960
	ds_load_b32 v31, v10 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v144, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v145, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v146, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v147, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v144, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v145, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v146, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v147, v31, v23
	s_cmp_le_u32 s18, 27
	s_cbranch_scc1 .Lbf16hc32_st6_cdone
	ds_load_b32 v24, v10 offset:12288
	ds_load_b32 v25, v10 offset:12416
	ds_load_b32 v26, v10 offset:12544
	ds_load_b32 v27, v10 offset:12672
	ds_load_b32 v28, v10 offset:12800
	ds_load_b32 v29, v10 offset:12928
	ds_load_b32 v30, v10 offset:13056
	ds_load_b32 v31, v10 offset:13184
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v148, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v149, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v150, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v151, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v148, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v149, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v150, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v151, v31, v23
	.Lbf16hc32_st6_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16hc32_st6_done:
	s_cmp_le_u32 s18, 28
	s_cbranch_scc1 .Lbf16hc32_st7_done
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 28
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st7_h0_nofill
	s_mul_i32 s29, s37, 28
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24
	ds_store_b32 v9, v25 offset:128
	ds_store_b32 v9, v26 offset:256
	ds_store_b32 v9, v27 offset:384
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:512
	ds_store_b32 v9, v29 offset:640
	ds_store_b32 v9, v30 offset:768
	ds_store_b32 v9, v31 offset:896
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st7_h0_nofill:
	s_lshr_b32 s28, s21, 2
	s_add_i32 s28, s28, 30
	s_cmp_ge_u32 s28, s18
	s_cbranch_scc1 .Lbf16hc32_st7_h1_nofill
	s_mul_i32 s29, s37, 28
	s_add_i32 s29, s29, s39
	v_add_nc_u32_e32 v13, s29, v3
	s_mov_b32 exec_lo, s32
	global_load_b128 v[24:27], v13, s[10:11]
	global_load_b128 v[28:31], v13, s[10:11] offset:16
	s_waitcnt vmcnt(1)
	ds_store_b32 v9, v24 offset:8192
	ds_store_b32 v9, v25 offset:8320
	ds_store_b32 v9, v26 offset:8448
	ds_store_b32 v9, v27 offset:8576
	s_waitcnt vmcnt(0)
	ds_store_b32 v9, v28 offset:8704
	ds_store_b32 v9, v29 offset:8832
	ds_store_b32 v9, v30 offset:8960
	ds_store_b32 v9, v31 offset:9088
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_st7_h1_nofill:
	s_barrier
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc32_st7_cdone
	s_mov_b32 exec_lo, s32
	s_cmp_le_u32 s18, 28
	s_cbranch_scc1 .Lbf16hc32_st7_cdone
	ds_load_b32 v24, v10
	ds_load_b32 v25, v10 offset:128
	ds_load_b32 v26, v10 offset:256
	ds_load_b32 v27, v10 offset:384
	ds_load_b32 v28, v10 offset:512
	ds_load_b32 v29, v10 offset:640
	ds_load_b32 v30, v10 offset:768
	ds_load_b32 v31, v10 offset:896
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v152, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v153, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v154, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v155, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v152, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v153, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v154, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v155, v31, v23
	s_cmp_le_u32 s18, 29
	s_cbranch_scc1 .Lbf16hc32_st7_cdone
	ds_load_b32 v24, v10 offset:4096
	ds_load_b32 v25, v10 offset:4224
	ds_load_b32 v26, v10 offset:4352
	ds_load_b32 v27, v10 offset:4480
	ds_load_b32 v28, v10 offset:4608
	ds_load_b32 v29, v10 offset:4736
	ds_load_b32 v30, v10 offset:4864
	ds_load_b32 v31, v10 offset:4992
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v156, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v157, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v158, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v159, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v156, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v157, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v158, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v159, v31, v23
	s_cmp_le_u32 s18, 30
	s_cbranch_scc1 .Lbf16hc32_st7_cdone
	ds_load_b32 v24, v10 offset:8192
	ds_load_b32 v25, v10 offset:8320
	ds_load_b32 v26, v10 offset:8448
	ds_load_b32 v27, v10 offset:8576
	ds_load_b32 v28, v10 offset:8704
	ds_load_b32 v29, v10 offset:8832
	ds_load_b32 v30, v10 offset:8960
	ds_load_b32 v31, v10 offset:9088
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v160, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v161, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v162, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v163, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v160, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v161, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v162, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v163, v31, v23
	s_cmp_le_u32 s18, 31
	s_cbranch_scc1 .Lbf16hc32_st7_cdone
	ds_load_b32 v24, v10 offset:12288
	ds_load_b32 v25, v10 offset:12416
	ds_load_b32 v26, v10 offset:12544
	ds_load_b32 v27, v10 offset:12672
	ds_load_b32 v28, v10 offset:12800
	ds_load_b32 v29, v10 offset:12928
	ds_load_b32 v30, v10 offset:13056
	ds_load_b32 v31, v10 offset:13184
	s_waitcnt lgkmcnt(7)
	v_fmac_f32_e32 v164, v24, v16
	s_waitcnt lgkmcnt(6)
	v_fmac_f32_e32 v165, v25, v17
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v166, v26, v18
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v167, v27, v19
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v164, v28, v20
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v165, v29, v21
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v166, v30, v22
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v167, v31, v23
	.Lbf16hc32_st7_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier
	.Lbf16hc32_st7_done:
	s_add_i32 s36, s36, 1
	s_cmp_lt_u32 s36, s27
	s_cbranch_scc1 .Lbf16hc32_tile
	.Lbf16hc32_fold:
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lbf16hc32_pdone
	v_add_f32_e32 v41, v41, v40
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v41
	v_add_f32_e32 v45, v45, v44
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v45
	v_add_f32_e32 v49, v49, v48
	v_add_f32_e32 v51, v51, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v49
	v_add_f32_e32 v53, v53, v52
	v_add_f32_e32 v55, v55, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v53
	v_add_f32_e32 v57, v57, v56
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v57
	v_add_f32_e32 v61, v61, v60
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v61
	v_add_f32_e32 v65, v65, v64
	v_add_f32_e32 v67, v67, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v65
	v_add_f32_e32 v69, v69, v68
	v_add_f32_e32 v71, v71, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v69
	v_add_f32_e32 v73, v73, v72
	v_add_f32_e32 v75, v75, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v75, v75, v73
	v_add_f32_e32 v77, v77, v76
	v_add_f32_e32 v79, v79, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v79, v79, v77
	v_add_f32_e32 v81, v81, v80
	v_add_f32_e32 v83, v83, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v83, v83, v81
	v_add_f32_e32 v85, v85, v84
	v_add_f32_e32 v87, v87, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v87, v87, v85
	v_add_f32_e32 v89, v89, v88
	v_add_f32_e32 v91, v91, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v91, v91, v89
	v_add_f32_e32 v93, v93, v92
	v_add_f32_e32 v95, v95, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v95, v95, v93
	v_add_f32_e32 v97, v97, v96
	v_add_f32_e32 v99, v99, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v99, v99, v97
	v_add_f32_e32 v101, v101, v100
	v_add_f32_e32 v103, v103, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v103, v103, v101
	v_add_f32_e32 v105, v105, v104
	v_add_f32_e32 v107, v107, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v107, v107, v105
	v_add_f32_e32 v109, v109, v108
	v_add_f32_e32 v111, v111, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v111, v111, v109
	v_add_f32_e32 v113, v113, v112
	v_add_f32_e32 v115, v115, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v115, v115, v113
	v_add_f32_e32 v117, v117, v116
	v_add_f32_e32 v119, v119, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v119, v119, v117
	v_add_f32_e32 v121, v121, v120
	v_add_f32_e32 v123, v123, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v123, v123, v121
	v_add_f32_e32 v125, v125, v124
	v_add_f32_e32 v127, v127, v126
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v127, v127, v125
	v_add_f32_e32 v129, v129, v128
	v_add_f32_e32 v131, v131, v130
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v131, v131, v129
	v_add_f32_e32 v133, v133, v132
	v_add_f32_e32 v135, v135, v134
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v135, v135, v133
	v_add_f32_e32 v137, v137, v136
	v_add_f32_e32 v139, v139, v138
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v139, v139, v137
	v_add_f32_e32 v141, v141, v140
	v_add_f32_e32 v143, v143, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v143, v143, v141
	v_add_f32_e32 v145, v145, v144
	v_add_f32_e32 v147, v147, v146
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v147, v147, v145
	v_add_f32_e32 v149, v149, v148
	v_add_f32_e32 v151, v151, v150
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v151, v151, v149
	v_add_f32_e32 v153, v153, v152
	v_add_f32_e32 v155, v155, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v155, v155, v153
	v_add_f32_e32 v157, v157, v156
	v_add_f32_e32 v159, v159, v158
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v159, v159, v157
	v_add_f32_e32 v161, v161, v160
	v_add_f32_e32 v163, v163, v162
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v163, v163, v161
	v_add_f32_e32 v165, v165, v164
	v_add_f32_e32 v167, v167, v166
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v167, v167, v165
	ds_swizzle_b32 v24, v43 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v47 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v51 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v55 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v59 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v63 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v71 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v75 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v79 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v83 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v87 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v36, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v37, v95 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v38, v99 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v39, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	ds_bpermute_b32 v25, v4, v47
	ds_bpermute_b32 v26, v4, v51
	ds_bpermute_b32 v27, v4, v55
	ds_bpermute_b32 v28, v4, v59
	ds_bpermute_b32 v29, v4, v63
	ds_bpermute_b32 v30, v4, v67
	ds_bpermute_b32 v31, v4, v71
	ds_bpermute_b32 v32, v4, v75
	ds_bpermute_b32 v33, v4, v79
	ds_bpermute_b32 v34, v4, v83
	ds_bpermute_b32 v35, v4, v87
	ds_bpermute_b32 v36, v4, v91
	ds_bpermute_b32 v37, v4, v95
	ds_bpermute_b32 v38, v4, v99
	ds_bpermute_b32 v39, v4, v103
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v43, v43, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v47, v47, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v51, v51, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v55, v55, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v59, v59, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v63, v63, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v67, v67, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v71, v71, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v79, v79, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v83, v83, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v87, v87, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v91, v91, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v95, v95, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v99, v99, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v103, v39
	ds_swizzle_b32 v24, v107 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v111 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v119 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v123 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v127 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v131 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v135 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v139 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v143 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v147 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v151 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v36, v155 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v37, v159 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v38, v163 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v39, v167 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v107, v107, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v111, v111, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v115, v115, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v119, v119, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v123, v123, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v127, v127, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v131, v131, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v135, v135, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v139, v139, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v143, v143, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v147, v147, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v151, v151, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v155, v155, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v159, v159, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v163, v163, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v167, v167, v39
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v107
	ds_bpermute_b32 v25, v4, v111
	ds_bpermute_b32 v26, v4, v115
	ds_bpermute_b32 v27, v4, v119
	ds_bpermute_b32 v28, v4, v123
	ds_bpermute_b32 v29, v4, v127
	ds_bpermute_b32 v30, v4, v131
	ds_bpermute_b32 v31, v4, v135
	ds_bpermute_b32 v32, v4, v139
	ds_bpermute_b32 v33, v4, v143
	ds_bpermute_b32 v34, v4, v147
	ds_bpermute_b32 v35, v4, v151
	ds_bpermute_b32 v36, v4, v155
	ds_bpermute_b32 v37, v4, v159
	ds_bpermute_b32 v38, v4, v163
	ds_bpermute_b32 v39, v4, v167
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v107, v107, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v111, v111, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v115, v115, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v119, v119, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v123, v123, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v127, v127, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v131, v131, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v135, v135, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v139, v139, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v143, v143, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v147, v147, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v151, v151, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v155, v155, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v159, v159, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v163, v163, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v167, v167, v39
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v107
	ds_bpermute_b32 v25, v4, v111
	ds_bpermute_b32 v26, v4, v115
	ds_bpermute_b32 v27, v4, v119
	ds_bpermute_b32 v28, v4, v123
	ds_bpermute_b32 v29, v4, v127
	ds_bpermute_b32 v30, v4, v131
	ds_bpermute_b32 v31, v4, v135
	ds_bpermute_b32 v32, v4, v139
	ds_bpermute_b32 v33, v4, v143
	ds_bpermute_b32 v34, v4, v147
	ds_bpermute_b32 v35, v4, v151
	ds_bpermute_b32 v36, v4, v155
	ds_bpermute_b32 v37, v4, v159
	ds_bpermute_b32 v38, v4, v163
	ds_bpermute_b32 v39, v4, v167
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v107, v107, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v111, v111, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v115, v115, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v119, v119, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v123, v123, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v127, v127, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v131, v131, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v135, v135, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v139, v139, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v143, v143, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v147, v147, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v151, v151, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v155, v155, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v159, v159, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v163, v163, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v167, v167, v39
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v107
	ds_bpermute_b32 v25, v4, v111
	ds_bpermute_b32 v26, v4, v115
	ds_bpermute_b32 v27, v4, v119
	ds_bpermute_b32 v28, v4, v123
	ds_bpermute_b32 v29, v4, v127
	ds_bpermute_b32 v30, v4, v131
	ds_bpermute_b32 v31, v4, v135
	ds_bpermute_b32 v32, v4, v139
	ds_bpermute_b32 v33, v4, v143
	ds_bpermute_b32 v34, v4, v147
	ds_bpermute_b32 v35, v4, v151
	ds_bpermute_b32 v36, v4, v155
	ds_bpermute_b32 v37, v4, v159
	ds_bpermute_b32 v38, v4, v163
	ds_bpermute_b32 v39, v4, v167
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v107, v107, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v111, v111, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v115, v115, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v119, v119, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v123, v123, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v127, v127, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v131, v131, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v135, v135, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v139, v139, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v143, v143, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v147, v147, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v151, v151, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v155, v155, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v159, v159, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v163, v163, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v167, v167, v39
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v107
	ds_bpermute_b32 v25, v4, v111
	ds_bpermute_b32 v26, v4, v115
	ds_bpermute_b32 v27, v4, v119
	ds_bpermute_b32 v28, v4, v123
	ds_bpermute_b32 v29, v4, v127
	ds_bpermute_b32 v30, v4, v131
	ds_bpermute_b32 v31, v4, v135
	ds_bpermute_b32 v32, v4, v139
	ds_bpermute_b32 v33, v4, v143
	ds_bpermute_b32 v34, v4, v147
	ds_bpermute_b32 v35, v4, v151
	ds_bpermute_b32 v36, v4, v155
	ds_bpermute_b32 v37, v4, v159
	ds_bpermute_b32 v38, v4, v163
	ds_bpermute_b32 v39, v4, v167
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v107, v107, v24
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v111, v111, v25
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v115, v115, v26
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v119, v119, v27
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v123, v123, v28
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v127, v127, v29
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v131, v131, v30
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v135, v135, v31
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v139, v139, v32
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v143, v143, v33
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v147, v147, v34
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v151, v151, v35
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v155, v155, v36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v159, v159, v37
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v163, v163, v38
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v167, v167, v39
	v_cmpx_eq_u32_e32 0, v5
	ds_store_b32 v15, v43
	s_cmp_le_u32 s18, 1
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v47 offset:16
	s_cmp_le_u32 s18, 2
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v51 offset:32
	s_cmp_le_u32 s18, 3
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v55 offset:48
	s_cmp_le_u32 s18, 4
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v59 offset:64
	s_cmp_le_u32 s18, 5
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v63 offset:80
	s_cmp_le_u32 s18, 6
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v67 offset:96
	s_cmp_le_u32 s18, 7
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v71 offset:112
	s_cmp_le_u32 s18, 8
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v75 offset:128
	s_cmp_le_u32 s18, 9
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v79 offset:144
	s_cmp_le_u32 s18, 10
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v83 offset:160
	s_cmp_le_u32 s18, 11
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v87 offset:176
	s_cmp_le_u32 s18, 12
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v91 offset:192
	s_cmp_le_u32 s18, 13
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v95 offset:208
	s_cmp_le_u32 s18, 14
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v99 offset:224
	s_cmp_le_u32 s18, 15
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v103 offset:240
	s_cmp_le_u32 s18, 16
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v107 offset:256
	s_cmp_le_u32 s18, 17
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v111 offset:272
	s_cmp_le_u32 s18, 18
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v115 offset:288
	s_cmp_le_u32 s18, 19
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v119 offset:304
	s_cmp_le_u32 s18, 20
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v123 offset:320
	s_cmp_le_u32 s18, 21
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v127 offset:336
	s_cmp_le_u32 s18, 22
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v131 offset:352
	s_cmp_le_u32 s18, 23
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v135 offset:368
	s_cmp_le_u32 s18, 24
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v139 offset:384
	s_cmp_le_u32 s18, 25
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v143 offset:400
	s_cmp_le_u32 s18, 26
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v147 offset:416
	s_cmp_le_u32 s18, 27
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v151 offset:432
	s_cmp_le_u32 s18, 28
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v155 offset:448
	s_cmp_le_u32 s18, 29
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v159 offset:464
	s_cmp_le_u32 s18, 30
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v163 offset:480
	s_cmp_le_u32 s18, 31
	s_cbranch_scc1 .Lbf16hc32_pstored
	ds_store_b32 v15, v167 offset:496
	.Lbf16hc32_pstored:
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_pdone:
	s_barrier
	s_add_i32 s45, s21, 0
	s_cmp_ge_u32 s45, s18
	s_cbranch_scc1 .Lbf16hc32_iskip0
	s_lshl_b32 s28, s45, 4
	v_mov_b32_e32 v40, s28
	ds_load_b128 v[41:44], v40
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s64, v41
	v_readfirstlane_b32 s65, v42
	v_readfirstlane_b32 s66, v43
	v_readfirstlane_b32 s67, v44
	s_add_f32 s69, s64, s65
	s_add_f32 s70, s66, s67
	s_delay_alu instid0(SALU_CYCLE_3)
	s_add_f32 s68, s69, s70
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lbf16hc32_item0_done
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s68, s16, s68
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s69, s68, 0xbfb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s70, s69, 0x80000000
	s_rndne_f32 s71, s69
	s_fmamk_f32 s70, s68, 0xbfb8aa3b, s70
	s_cmp_ngt_f32 s68, 0x42ce8ed0
	s_sub_f32 s69, s69, s71
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s70, s68, 0xb2a5705f, s70
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_nlt_f32 s68, 0xc2b17218
	s_add_f32 s69, s69, s70
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v24, s69
	s_cvt_i32_f32 s69, s71
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, s69
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s68, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s68, v24
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	.Lbf16hc32_item0_done:
	s_mul_i32 s46, s45, s38
	s_add_i32 s28, s44, 0
	s_lshl_b32 s28, s28, 2
	s_add_i32 s46, s46, s28
	v_mov_b32_e32 v45, s46
	v_mov_b32_e32 v46, s68
	s_mov_b32 exec_lo, 1
	global_store_b32 v45, v46, s[12:13]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_iskip0:
	s_add_i32 s45, s21, 8
	s_cmp_ge_u32 s45, s18
	s_cbranch_scc1 .Lbf16hc32_iskip1
	s_lshl_b32 s28, s45, 4
	v_mov_b32_e32 v40, s28
	ds_load_b128 v[41:44], v40
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s64, v41
	v_readfirstlane_b32 s65, v42
	v_readfirstlane_b32 s66, v43
	v_readfirstlane_b32 s67, v44
	s_add_f32 s69, s64, s65
	s_add_f32 s70, s66, s67
	s_delay_alu instid0(SALU_CYCLE_3)
	s_add_f32 s68, s69, s70
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lbf16hc32_item1_done
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s68, s16, s68
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s69, s68, 0xbfb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s70, s69, 0x80000000
	s_rndne_f32 s71, s69
	s_fmamk_f32 s70, s68, 0xbfb8aa3b, s70
	s_cmp_ngt_f32 s68, 0x42ce8ed0
	s_sub_f32 s69, s69, s71
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s70, s68, 0xb2a5705f, s70
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_nlt_f32 s68, 0xc2b17218
	s_add_f32 s69, s69, s70
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v24, s69
	s_cvt_i32_f32 s69, s71
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, s69
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s68, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s68, v24
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	.Lbf16hc32_item1_done:
	s_mul_i32 s46, s45, s38
	s_add_i32 s28, s44, 0
	s_lshl_b32 s28, s28, 2
	s_add_i32 s46, s46, s28
	v_mov_b32_e32 v45, s46
	v_mov_b32_e32 v46, s68
	s_mov_b32 exec_lo, 1
	global_store_b32 v45, v46, s[12:13]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_iskip1:
	s_add_i32 s45, s21, 16
	s_cmp_ge_u32 s45, s18
	s_cbranch_scc1 .Lbf16hc32_iskip2
	s_lshl_b32 s28, s45, 4
	v_mov_b32_e32 v40, s28
	ds_load_b128 v[41:44], v40
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s64, v41
	v_readfirstlane_b32 s65, v42
	v_readfirstlane_b32 s66, v43
	v_readfirstlane_b32 s67, v44
	s_add_f32 s69, s64, s65
	s_add_f32 s70, s66, s67
	s_delay_alu instid0(SALU_CYCLE_3)
	s_add_f32 s68, s69, s70
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lbf16hc32_item2_done
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s68, s16, s68
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s69, s68, 0xbfb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s70, s69, 0x80000000
	s_rndne_f32 s71, s69
	s_fmamk_f32 s70, s68, 0xbfb8aa3b, s70
	s_cmp_ngt_f32 s68, 0x42ce8ed0
	s_sub_f32 s69, s69, s71
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s70, s68, 0xb2a5705f, s70
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_nlt_f32 s68, 0xc2b17218
	s_add_f32 s69, s69, s70
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v24, s69
	s_cvt_i32_f32 s69, s71
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, s69
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s68, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s68, v24
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	.Lbf16hc32_item2_done:
	s_mul_i32 s46, s45, s38
	s_add_i32 s28, s44, 0
	s_lshl_b32 s28, s28, 2
	s_add_i32 s46, s46, s28
	v_mov_b32_e32 v45, s46
	v_mov_b32_e32 v46, s68
	s_mov_b32 exec_lo, 1
	global_store_b32 v45, v46, s[12:13]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_iskip2:
	s_add_i32 s45, s21, 24
	s_cmp_ge_u32 s45, s18
	s_cbranch_scc1 .Lbf16hc32_iskip3
	s_lshl_b32 s28, s45, 4
	v_mov_b32_e32 v40, s28
	ds_load_b128 v[41:44], v40
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s64, v41
	v_readfirstlane_b32 s65, v42
	v_readfirstlane_b32 s66, v43
	v_readfirstlane_b32 s67, v44
	s_add_f32 s69, s64, s65
	s_add_f32 s70, s66, s67
	s_delay_alu instid0(SALU_CYCLE_3)
	s_add_f32 s68, s69, s70
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lbf16hc32_item3_done
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s68, s16, s68
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s69, s68, 0xbfb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s70, s69, 0x80000000
	s_rndne_f32 s71, s69
	s_fmamk_f32 s70, s68, 0xbfb8aa3b, s70
	s_cmp_ngt_f32 s68, 0x42ce8ed0
	s_sub_f32 s69, s69, s71
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s70, s68, 0xb2a5705f, s70
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_nlt_f32 s68, 0xc2b17218
	s_add_f32 s69, s69, s70
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v24, s69
	s_cvt_i32_f32 s69, s71
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, s69
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s68, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s68, v24
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	.Lbf16hc32_item3_done:
	s_mul_i32 s46, s45, s38
	s_add_i32 s28, s44, 0
	s_lshl_b32 s28, s28, 2
	s_add_i32 s46, s46, s28
	v_mov_b32_e32 v45, s46
	v_mov_b32_e32 v46, s68
	s_mov_b32 exec_lo, 1
	global_store_b32 v45, v46, s[12:13]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_iskip3:
	s_add_i32 s45, s21, 0
	s_cmp_ge_u32 s45, s18
	s_cbranch_scc1 .Lbf16hc32_iskip4
	s_add_i32 s28, s44, 1
	s_cmp_ge_u32 s28, s14
	s_cbranch_scc1 .Lbf16hc32_iskip4
	s_lshl_b32 s28, s45, 4
	s_add_i32 s28, s28, 0x200
	v_mov_b32_e32 v40, s28
	ds_load_b128 v[41:44], v40
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s64, v41
	v_readfirstlane_b32 s65, v42
	v_readfirstlane_b32 s66, v43
	v_readfirstlane_b32 s67, v44
	s_add_f32 s69, s64, s65
	s_add_f32 s70, s66, s67
	s_delay_alu instid0(SALU_CYCLE_3)
	s_add_f32 s68, s69, s70
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lbf16hc32_item4_done
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s68, s16, s68
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s69, s68, 0xbfb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s70, s69, 0x80000000
	s_rndne_f32 s71, s69
	s_fmamk_f32 s70, s68, 0xbfb8aa3b, s70
	s_cmp_ngt_f32 s68, 0x42ce8ed0
	s_sub_f32 s69, s69, s71
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s70, s68, 0xb2a5705f, s70
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_nlt_f32 s68, 0xc2b17218
	s_add_f32 s69, s69, s70
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v24, s69
	s_cvt_i32_f32 s69, s71
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, s69
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s68, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s68, v24
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	.Lbf16hc32_item4_done:
	s_mul_i32 s46, s45, s38
	s_add_i32 s28, s44, 1
	s_lshl_b32 s28, s28, 2
	s_add_i32 s46, s46, s28
	v_mov_b32_e32 v45, s46
	v_mov_b32_e32 v46, s68
	s_mov_b32 exec_lo, 1
	global_store_b32 v45, v46, s[12:13]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_iskip4:
	s_add_i32 s45, s21, 8
	s_cmp_ge_u32 s45, s18
	s_cbranch_scc1 .Lbf16hc32_iskip5
	s_add_i32 s28, s44, 1
	s_cmp_ge_u32 s28, s14
	s_cbranch_scc1 .Lbf16hc32_iskip5
	s_lshl_b32 s28, s45, 4
	s_add_i32 s28, s28, 0x200
	v_mov_b32_e32 v40, s28
	ds_load_b128 v[41:44], v40
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s64, v41
	v_readfirstlane_b32 s65, v42
	v_readfirstlane_b32 s66, v43
	v_readfirstlane_b32 s67, v44
	s_add_f32 s69, s64, s65
	s_add_f32 s70, s66, s67
	s_delay_alu instid0(SALU_CYCLE_3)
	s_add_f32 s68, s69, s70
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lbf16hc32_item5_done
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s68, s16, s68
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s69, s68, 0xbfb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s70, s69, 0x80000000
	s_rndne_f32 s71, s69
	s_fmamk_f32 s70, s68, 0xbfb8aa3b, s70
	s_cmp_ngt_f32 s68, 0x42ce8ed0
	s_sub_f32 s69, s69, s71
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s70, s68, 0xb2a5705f, s70
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_nlt_f32 s68, 0xc2b17218
	s_add_f32 s69, s69, s70
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v24, s69
	s_cvt_i32_f32 s69, s71
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, s69
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s68, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s68, v24
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	.Lbf16hc32_item5_done:
	s_mul_i32 s46, s45, s38
	s_add_i32 s28, s44, 1
	s_lshl_b32 s28, s28, 2
	s_add_i32 s46, s46, s28
	v_mov_b32_e32 v45, s46
	v_mov_b32_e32 v46, s68
	s_mov_b32 exec_lo, 1
	global_store_b32 v45, v46, s[12:13]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_iskip5:
	s_add_i32 s45, s21, 16
	s_cmp_ge_u32 s45, s18
	s_cbranch_scc1 .Lbf16hc32_iskip6
	s_add_i32 s28, s44, 1
	s_cmp_ge_u32 s28, s14
	s_cbranch_scc1 .Lbf16hc32_iskip6
	s_lshl_b32 s28, s45, 4
	s_add_i32 s28, s28, 0x200
	v_mov_b32_e32 v40, s28
	ds_load_b128 v[41:44], v40
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s64, v41
	v_readfirstlane_b32 s65, v42
	v_readfirstlane_b32 s66, v43
	v_readfirstlane_b32 s67, v44
	s_add_f32 s69, s64, s65
	s_add_f32 s70, s66, s67
	s_delay_alu instid0(SALU_CYCLE_3)
	s_add_f32 s68, s69, s70
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lbf16hc32_item6_done
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s68, s16, s68
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s69, s68, 0xbfb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s70, s69, 0x80000000
	s_rndne_f32 s71, s69
	s_fmamk_f32 s70, s68, 0xbfb8aa3b, s70
	s_cmp_ngt_f32 s68, 0x42ce8ed0
	s_sub_f32 s69, s69, s71
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s70, s68, 0xb2a5705f, s70
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_nlt_f32 s68, 0xc2b17218
	s_add_f32 s69, s69, s70
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v24, s69
	s_cvt_i32_f32 s69, s71
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, s69
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s68, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s68, v24
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	.Lbf16hc32_item6_done:
	s_mul_i32 s46, s45, s38
	s_add_i32 s28, s44, 1
	s_lshl_b32 s28, s28, 2
	s_add_i32 s46, s46, s28
	v_mov_b32_e32 v45, s46
	v_mov_b32_e32 v46, s68
	s_mov_b32 exec_lo, 1
	global_store_b32 v45, v46, s[12:13]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_iskip6:
	s_add_i32 s45, s21, 24
	s_cmp_ge_u32 s45, s18
	s_cbranch_scc1 .Lbf16hc32_iskip7
	s_add_i32 s28, s44, 1
	s_cmp_ge_u32 s28, s14
	s_cbranch_scc1 .Lbf16hc32_iskip7
	s_lshl_b32 s28, s45, 4
	s_add_i32 s28, s28, 0x200
	v_mov_b32_e32 v40, s28
	ds_load_b128 v[41:44], v40
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s64, v41
	v_readfirstlane_b32 s65, v42
	v_readfirstlane_b32 s66, v43
	v_readfirstlane_b32 s67, v44
	s_add_f32 s69, s64, s65
	s_add_f32 s70, s66, s67
	s_delay_alu instid0(SALU_CYCLE_3)
	s_add_f32 s68, s69, s70
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lbf16hc32_item7_done
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s68, s16, s68
	s_delay_alu instid0(SALU_CYCLE_3)
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	s_mul_f32 s69, s68, 0xbfb8aa3b
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_xor_b32 s70, s69, 0x80000000
	s_rndne_f32 s71, s69
	s_fmamk_f32 s70, s68, 0xbfb8aa3b, s70
	s_cmp_ngt_f32 s68, 0x42ce8ed0
	s_sub_f32 s69, s69, s71
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_fmamk_f32 s70, s68, 0xb2a5705f, s70
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_nlt_f32 s68, 0xc2b17218
	s_add_f32 s69, s69, s70
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v24, s69
	s_cvt_i32_f32 s69, s71
	s_delay_alu instid0(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, s69
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s68, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s68, v24
	s_bfe_u32 s69, s68, 0x10010
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s69, s68, s69
	s_addk_i32 s69, 0x7fff
	s_and_b32 s70, s68, 0xffff
	s_or_b32 s71, s68, 0x10000
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s70, s68, s71
	s_and_b32 s71, s68, 0x7f800000
	s_cmp_eq_u32 s71, 0x7f800000
	s_cselect_b32 s69, s70, s69
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s68, s69, 0xffff0000
	.Lbf16hc32_item7_done:
	s_mul_i32 s46, s45, s38
	s_add_i32 s28, s44, 1
	s_lshl_b32 s28, s28, 2
	s_add_i32 s46, s46, s28
	v_mov_b32_e32 v45, s46
	v_mov_b32_e32 v46, s68
	s_mov_b32 exec_lo, 1
	global_store_b32 v45, v46, s[12:13]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16hc32_iskip7:
	s_barrier
	.Lbf16hc32_end:
	s_endpgm
.Lgemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151_end:
.size gemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151, .Lgemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151_end-gemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151
	.amdhsa_group_segment_fixed_size 16384
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 48
	.amdhsa_user_sgpr_count 2
	.amdhsa_user_sgpr_dispatch_ptr 0
	.amdhsa_user_sgpr_queue_ptr 0
	.amdhsa_user_sgpr_kernarg_segment_ptr 1
	.amdhsa_user_sgpr_dispatch_id 0
	.amdhsa_user_sgpr_private_segment_size 0
	.amdhsa_wavefront_size32 1
	.amdhsa_uses_dynamic_stack 0
	.amdhsa_enable_private_segment 0
	.amdhsa_system_sgpr_workgroup_id_x 1
	.amdhsa_system_sgpr_workgroup_id_y 0
	.amdhsa_system_sgpr_workgroup_id_z 0
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 168
	.amdhsa_next_free_sgpr 80
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151_end-gemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A0
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A1
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A2
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A3
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y2
        .offset: 56
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y3
        .offset: 64
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M0
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: M1
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: M2
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: M3
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: R
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: gemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151
    .private_segment_fixed_size: 0
    .sgpr_count: 66
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 104
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A0
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A1
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A2
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A3
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y2
        .offset: 56
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y3
        .offset: 64
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M0
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: M1
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: M2
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: M3
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: R
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: gemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151
    .private_segment_fixed_size: 0
    .sgpr_count: 66
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 168
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: W
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: X
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: scale
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: hc_act
        .offset: 36
        .size: 4
        .value_kind: by_value
      - 
        .name: R
        .offset: 40
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 256
    .name: gemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151
    .private_segment_fixed_size: 0
    .sgpr_count: 82
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 104
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: W
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: X
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: scale
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: hc_act
        .offset: 36
        .size: 4
        .value_kind: by_value
      - 
        .name: R
        .offset: 40
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 256
    .name: gemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151
    .private_segment_fixed_size: 0
    .sgpr_count: 82
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 168
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1151
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
