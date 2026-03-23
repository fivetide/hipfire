	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1010"
	.protected	attention_hfq4_kv       ; -- Begin function attention_hfq4_kv
	.globl	attention_hfq4_kv
	.p2align	8
	.type	attention_hfq4_kv,@function
attention_hfq4_kv:                      ; @attention_hfq4_kv
; %bb.0:
	s_load_dwordx4 s[0:3], s[4:5], 0x28
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_i32 s6, s0
	s_cbranch_scc1 .LBB0_36
; %bb.1:
	s_ashr_i32 s3, s1, 31
	s_ashr_i32 s10, s0, 31
	s_add_i32 s7, s1, s3
	s_add_i32 s0, s0, s10
	s_xor_b32 s7, s7, s3
	s_xor_b32 s0, s0, s10
	v_cvt_f32_u32_e32 v1, s7
	s_sub_i32 s9, 0, s7
	s_xor_b32 s3, s10, s3
	s_clause 0x1
	s_load_dwordx2 s[12:13], s[4:5], 0x20
	s_load_dword s14, s[4:5], 0x4c
	v_rcp_iflag_f32_e32 v1, v1
	v_mov_b32_e32 v7, 0xf149f2ca
	s_mov_b32 s20, 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s8, v1
	s_mul_i32 s9, s9, s8
	s_mul_hi_u32 s9, s8, s9
	s_add_i32 s8, s8, s9
	s_mul_hi_u32 s8, s0, s8
	s_mul_i32 s9, s8, s7
	s_sub_i32 s0, s0, s9
	s_add_i32 s9, s8, 1
	s_sub_i32 s10, s0, s7
	s_cmp_ge_u32 s0, s7
	s_cselect_b32 s8, s9, s8
	s_cselect_b32 s0, s10, s0
	s_add_i32 s9, s8, 1
	s_cmp_ge_u32 s0, s7
	s_cselect_b32 s0, s9, s8
	s_ashr_i32 s15, s6, 31
	s_xor_b32 s0, s0, s3
	s_add_i32 s9, s6, s15
	s_sub_i32 s0, s0, s3
	s_xor_b32 s16, s9, s15
	s_ashr_i32 s3, s0, 31
	s_add_i32 s0, s0, s3
	s_xor_b32 s0, s0, s3
	v_cvt_f32_u32_e32 v1, s0
	s_sub_i32 s8, 0, s0
	v_rcp_iflag_f32_e32 v1, v1
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s7, v1
	s_mul_i32 s8, s8, s7
	s_mul_hi_u32 s8, s7, s8
	s_add_i32 s7, s7, s8
	s_load_dwordx4 s[8:11], s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	s_load_dword s18, s[12:13], 0x0
	s_mul_hi_u32 s7, s16, s7
	s_xor_b32 s13, s15, s3
	s_mul_i32 s12, s7, s0
	s_sub_i32 s3, s16, s12
	s_add_i32 s12, s7, 1
	s_sub_i32 s15, s3, s0
	s_cmp_ge_u32 s3, s0
	s_cselect_b32 s7, s12, s7
	s_cselect_b32 s3, s15, s3
	s_add_i32 s15, s7, 1
	s_cmp_ge_u32 s3, s0
	s_mul_i32 s12, s6, s2
	s_cselect_b32 s0, s15, s7
	s_lshr_b32 s6, s2, 31
	s_xor_b32 s0, s0, s13
	s_add_i32 s6, s2, s6
	s_sub_i32 s7, s0, s13
	s_ashr_i32 s19, s6, 1
	s_waitcnt lgkmcnt(0)
	v_cmp_ge_i32_e64 s0, s18, v0
	s_add_i32 s19, s19, 8
	s_and_b32 s3, s14, 0xffff
	s_ashr_i32 s13, s12, 31
	s_mul_i32 s6, s19, s1
	s_and_saveexec_b32 s21, s0
	s_cbranch_execz .LBB0_9
; %bb.2:                                ; %.lr.ph168
	s_clause 0x1
	s_load_dwordx4 s[28:31], s[4:5], 0x0
	s_load_dword s22, s[4:5], 0x38
	s_mul_i32 s1, s7, s19
	s_ashr_i32 s23, s6, 31
	s_ashr_i32 s5, s1, 31
	v_mov_b32_e32 v7, 0xf149f2ca
	v_mov_b32_e32 v8, v0
	s_mul_hi_i32 s24, s6, s3
	s_mul_i32 s25, s6, s3
	s_waitcnt lgkmcnt(0)
	s_add_u32 s4, s30, s1
	s_addc_u32 s5, s31, s5
	s_cmp_gt_i32 s2, 0
	v_mad_i64_i32 v[1:2], s1, s6, v0, s[4:5]
	s_cselect_b32 s26, -1, 0
	s_lshl_b64 s[14:15], s[12:13], 2
	s_add_u32 s1, s14, s28
	s_addc_u32 s15, s15, s29
	s_add_u32 s14, s1, 4
	v_add_co_u32 v1, vcc_lo, v1, 8
	v_add_co_ci_u32_e32 v2, vcc_lo, 0, v2, vcc_lo
	s_addc_u32 s15, s15, 0
	s_branch .LBB0_5
.LBB0_3:                                ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v9, 0
.LBB0_4:                                ; %._crit_edge
                                        ;   in Loop: Header=BB0_5 Depth=1
	v_lshl_add_u32 v4, v8, 2, 0
	v_add_nc_u32_e32 v8, s3, v8
	v_mul_f32_e32 v3, s22, v9
	v_max_f32_e32 v5, v7, v7
	v_add_co_u32 v1, s1, v1, s25
	v_cmp_lt_i32_e32 vcc_lo, s18, v8
	v_add_co_ci_u32_e64 v2, s1, s24, v2, s1
	v_max_f32_e32 v7, v5, v3
	s_waitcnt_vscnt null, 0x0
	ds_write_b32 v4, v3
	s_or_b32 s20, vcc_lo, s20
	s_waitcnt_depctr 0xffe3
	s_andn2_b32 exec_lo, exec_lo, s20
	s_cbranch_execz .LBB0_8
.LBB0_5:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_7 Depth 2
	s_andn2_b32 vcc_lo, exec_lo, s26
	s_cbranch_vccnz .LBB0_3
; %bb.6:                                ; %.lr.ph.preheader
                                        ;   in Loop: Header=BB0_5 Depth=1
	v_mad_u64_u32 v[3:4], s1, v8, s6, s[4:5]
	v_mov_b32_e32 v9, 0
	s_mov_b64 s[16:17], s[14:15]
	v_mad_u64_u32 v[4:5], s1, v8, s23, v[4:5]
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v5, v1
	s_mov_b32 s1, 0
	s_waitcnt_vscnt null, 0x0
	global_load_dwordx2 v[3:4], v[3:4], off
.LBB0_7:                                ; %.lr.ph
                                        ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_ubyte v10, v[5:6], off
	s_add_u32 s28, s16, -4
	s_addc_u32 s29, s17, -1
	v_add_co_u32 v5, vcc_lo, v5, 1
	s_load_dwordx2 s[28:29], s[28:29], 0x0
	v_add_co_ci_u32_e32 v6, vcc_lo, 0, v6, vcc_lo
	s_add_i32 s1, s1, 2
	s_add_u32 s16, s16, 8
	s_addc_u32 s17, s17, 0
	s_cmp_ge_i32 s1, s2
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v10, 15, v10
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v10, v10
	v_fma_f32 v11, v3, v11, v4
	v_fma_f32 v10, v3, v10, v4
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v11, s29, v11
	v_fmac_f32_e32 v11, s28, v10
	v_add_f32_e32 v9, v9, v11
	s_cbranch_scc0 .LBB0_7
	s_branch .LBB0_4
.LBB0_8:                                ; %Flow317
	s_or_b32 exec_lo, exec_lo, s20
.LBB0_9:                                ; %Flow318
	s_or_b32 exec_lo, exec_lo, s21
	s_lshl_b32 s1, s18, 2
	s_lshr_b32 s4, s3, 1
	s_add_i32 s1, s1, 0
	s_cmp_gt_u32 s3, 1
	v_lshl_add_u32 v2, v0, 2, s1
	s_cselect_b32 s5, -1, 0
	s_cmp_lt_u32 s3, 2
	v_add_nc_u32_e32 v1, 4, v2
	ds_write_b32 v2, v7 offset:4
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_cbranch_scc1 .LBB0_14
; %bb.10:
	s_mov_b32 s14, s4
	s_branch .LBB0_12
.LBB0_11:                               ;   in Loop: Header=BB0_12 Depth=1
	s_waitcnt_depctr 0xffe3
	s_or_b32 exec_lo, exec_lo, s15
	s_lshr_b32 s15, s14, 1
	s_cmp_lt_u32 s14, 2
	s_mov_b32 s14, s15
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt_vscnt null, 0x0
	buffer_gl0_inv
	s_cbranch_scc1 .LBB0_14
.LBB0_12:                               ; %.lr.ph176
                                        ; =>This Inner Loop Header: Depth=1
	v_cmp_gt_u32_e32 vcc_lo, s14, v0
	s_and_saveexec_b32 s15, vcc_lo
	s_cbranch_execz .LBB0_11
; %bb.13:                               ;   in Loop: Header=BB0_12 Depth=1
	v_lshl_add_u32 v2, s14, 2, v1
	s_waitcnt_vscnt null, 0x0
	ds_read_b32 v2, v2
	ds_read_b32 v3, v1
	s_waitcnt lgkmcnt(1)
	v_max_f32_e32 v2, v2, v2
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v3, v3, v3
	v_max_f32_e32 v2, v3, v2
	ds_write_b32 v1, v2
	s_branch .LBB0_11
.LBB0_14:                               ; %._crit_edge177
	v_mov_b32_e32 v2, s1
	s_waitcnt_vscnt null, 0x0
	ds_read_b32 v3, v2 offset:4
	v_mov_b32_e32 v2, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_and_saveexec_b32 s14, s0
	s_cbranch_execz .LBB0_18
; %bb.15:                               ; %.lr.ph182.preheader
	v_lshl_add_u32 v4, v0, 2, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v5, v0
	s_mov_b32 s15, 0
	s_lshl_b32 s16, s3, 2
.LBB0_16:                               ; %.lr.ph182
                                        ; =>This Inner Loop Header: Depth=1
	s_waitcnt_vscnt null, 0x0
	ds_read_b32 v6, v4
	v_add_nc_u32_e32 v5, s3, v5
	s_waitcnt lgkmcnt(0)
	v_sub_f32_e32 v6, v6, v3
	v_mul_f32_e32 v7, 0x3fb8aa3b, v6
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v6
	v_fma_f32 v8, 0x3fb8aa3b, v6, -v7
	v_rndne_f32_e32 v9, v7
	v_fmac_f32_e32 v8, 0x32a5705f, v6
	v_sub_f32_e32 v7, v7, v9
	v_add_f32_e32 v7, v7, v8
	v_cvt_i32_f32_e32 v8, v9
	v_exp_f32_e32 v7, v7
	v_ldexp_f32 v7, v7, v8
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v6
	v_cndmask_b32_e32 v6, 0x7f800000, v7, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, s18, v5
	ds_write_b32 v4, v6
	v_add_f32_e32 v2, v2, v6
	v_add_nc_u32_e32 v4, s16, v4
	s_or_b32 s15, vcc_lo, s15
	s_andn2_b32 exec_lo, exec_lo, s15
	s_cbranch_execnz .LBB0_16
; %bb.17:                               ; %Flow311
	s_or_b32 exec_lo, exec_lo, s15
.LBB0_18:                               ; %Flow312
	s_waitcnt_depctr 0xffe3
	s_or_b32 exec_lo, exec_lo, s14
	s_andn2_b32 vcc_lo, exec_lo, s5
	s_waitcnt_vscnt null, 0x0
	ds_write_b32 v1, v2
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_cbranch_vccz .LBB0_25
.LBB0_19:                               ; %._crit_edge189
	v_mov_b32_e32 v1, s1
	s_waitcnt_vscnt null, 0x0
	ds_read_b32 v1, v1 offset:4
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_22
; %bb.20:                               ; %.lr.ph193.preheader
	v_lshl_add_u32 v2, v0, 2, 0
	v_mov_b32_e32 v3, v0
	s_mov_b32 s0, 0
	s_lshl_b32 s4, s3, 2
.LBB0_21:                               ; %.lr.ph193
                                        ; =>This Inner Loop Header: Depth=1
	s_waitcnt_vscnt null, 0x0
	ds_read_b32 v4, v2
	v_add_nc_u32_e32 v3, s3, v3
	s_waitcnt lgkmcnt(0)
	v_div_scale_f32 v5, s5, v1, v1, v4
	v_div_scale_f32 v8, vcc_lo, v4, v1, v4
	v_rcp_f32_e32 v6, v5
	v_fma_f32 v7, -v5, v6, 1.0
	v_fmac_f32_e32 v6, v7, v6
	v_mul_f32_e32 v7, v8, v6
	v_fma_f32 v9, -v5, v7, v8
	v_fmac_f32_e32 v7, v9, v6
	v_fma_f32 v5, -v5, v7, v8
	v_div_fmas_f32 v5, v5, v6, v7
	v_cmp_lt_i32_e32 vcc_lo, s18, v3
	v_div_fixup_f32 v4, v5, v1, v4
	s_or_b32 s0, vcc_lo, s0
	ds_write_b32 v2, v4
	v_add_nc_u32_e32 v2, s4, v2
	s_andn2_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_21
.LBB0_22:                               ; %Flow308
	s_waitcnt_depctr 0xffe3
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_gt_i32_e32 vcc_lo, s2, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt_vscnt null, 0x0
	buffer_gl0_inv
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_36
; %bb.23:                               ; %.lr.ph206
	s_lshl_b64 s[4:5], s[12:13], 2
	s_mul_i32 s0, s7, s19
	s_add_u32 s1, s10, s4
	s_addc_u32 s14, s11, s5
	s_cmp_gt_i32 s18, -1
	v_mov_b32_e32 v1, 0
	s_cselect_b32 s15, -1, 0
	s_ashr_i32 s7, s6, 31
	s_ashr_i32 s5, s0, 31
	s_add_u32 s4, s8, s0
	s_addc_u32 s5, s9, s5
	s_add_i32 s0, s18, 1
	s_mul_i32 s21, s6, 3
	s_and_b32 s16, s0, 3
	s_cmp_gt_u32 s18, 2
	s_mul_hi_i32 s22, s6, 3
	s_cselect_b32 s17, -1, 0
	s_and_b32 s18, s0, -4
	s_cmp_lg_u32 s16, 0
	v_mov_b32_e32 v3, v1
	s_cselect_b32 s20, -1, 0
	s_add_u32 s23, s21, 8
	s_addc_u32 s24, s22, 0
	s_lshl_b64 s[8:9], s[6:7], 1
	s_lshl_b64 s[10:11], s[6:7], 2
	s_add_u32 s25, s8, 8
	v_mov_b32_e32 v2, v0
	s_addc_u32 s26, s9, 0
	s_add_u32 s27, s6, 8
	s_mov_b32 s19, 0
	s_addc_u32 s28, s7, 0
	s_branch .LBB0_28
.LBB0_24:                               ;   in Loop: Header=BB0_25 Depth=1
	s_waitcnt_depctr 0xffe3
	s_or_b32 exec_lo, exec_lo, s5
	s_lshr_b32 s5, s4, 1
	s_cmp_lt_u32 s4, 2
	s_mov_b32 s4, s5
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt_vscnt null, 0x0
	buffer_gl0_inv
	s_cbranch_scc1 .LBB0_19
.LBB0_25:                               ; %.lr.ph188
                                        ; =>This Inner Loop Header: Depth=1
	v_cmp_gt_u32_e32 vcc_lo, s4, v0
	s_and_saveexec_b32 s5, vcc_lo
	s_cbranch_execz .LBB0_24
; %bb.26:                               ;   in Loop: Header=BB0_25 Depth=1
	v_lshl_add_u32 v2, s4, 2, v1
	s_waitcnt_vscnt null, 0x0
	ds_read_b32 v2, v2
	ds_read_b32 v3, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v3
	ds_write_b32 v1, v2
	s_branch .LBB0_24
.LBB0_27:                               ; %._crit_edge200
                                        ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b32_e32 v0, v2
	v_add_co_u32 v2, vcc_lo, v2, s3
	v_add_co_ci_u32_e32 v3, vcc_lo, 0, v3, vcc_lo
	v_lshlrev_b64 v[5:6], 2, v[0:1]
	v_cmp_le_i32_e32 vcc_lo, s2, v2
	s_or_b32 s19, vcc_lo, s19
	v_add_co_u32 v5, s0, s1, v5
	v_add_co_ci_u32_e64 v6, s0, s14, v6, s0
	s_waitcnt_vscnt null, 0x0
	global_store_dword v[5:6], v4, off
	s_waitcnt_depctr 0xffe3
	s_andn2_b32 exec_lo, exec_lo, s19
	s_cbranch_execz .LBB0_36
.LBB0_28:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_31 Depth 2
                                        ;     Child Loop BB0_35 Depth 2
	v_mov_b32_e32 v4, 0
	s_andn2_b32 vcc_lo, exec_lo, s15
	s_cbranch_vccnz .LBB0_27
; %bb.29:                               ; %.lr.ph199
                                        ;   in Loop: Header=BB0_28 Depth=1
	v_lshrrev_b64 v[4:5], 1, v[2:3]
	v_and_b32_e32 v7, 1, v2
	s_mov_b32 s12, 0
	v_cmp_eq_u32_e64 s0, 0, v7
	v_add_co_u32 v0, vcc_lo, v4, 8
	v_add_co_ci_u32_e32 v6, vcc_lo, 0, v5, vcc_lo
	s_andn2_b32 vcc_lo, exec_lo, s17
	s_cbranch_vccnz .LBB0_33
; %bb.30:                               ; %.lr.ph199.new.preheader
                                        ;   in Loop: Header=BB0_28 Depth=1
	v_add_co_u32 v7, vcc_lo, s23, v4
	v_add_co_ci_u32_e32 v8, vcc_lo, s24, v5, vcc_lo
	v_add_co_u32 v9, vcc_lo, s25, v4
	v_add_co_ci_u32_e32 v10, vcc_lo, s26, v5, vcc_lo
	v_add_co_u32 v11, vcc_lo, s27, v4
	v_add_co_ci_u32_e32 v5, vcc_lo, s28, v5, vcc_lo
	v_mov_b32_e32 v4, 0
	s_mov_b32 s29, 0
	s_mov_b32 s30, 0
	s_mov_b64 s[12:13], s[4:5]
.LBB0_31:                               ; %.lr.ph199.new
                                        ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_co_u32 v12, vcc_lo, s12, v0
	v_add_co_ci_u32_e32 v13, vcc_lo, s13, v6, vcc_lo
	v_add_co_u32 v14, vcc_lo, s12, v11
	v_add_co_ci_u32_e32 v15, vcc_lo, s13, v5, vcc_lo
	s_waitcnt_vscnt null, 0x0
	global_load_ubyte v20, v[12:13], off
	v_add_co_u32 v16, vcc_lo, s12, v9
	global_load_ubyte v14, v[14:15], off
	v_add_co_ci_u32_e32 v17, vcc_lo, s13, v10, vcc_lo
	v_add_co_u32 v18, vcc_lo, s12, v7
	v_add_co_ci_u32_e32 v19, vcc_lo, s13, v8, vcc_lo
	global_load_ubyte v16, v[16:17], off
	s_load_dwordx2 s[34:35], s[12:13], 0x0
	v_mov_b32_e32 v15, s30
	global_load_ubyte v17, v[18:19], off
	s_add_u32 s36, s12, s6
	s_addc_u32 s37, s13, s7
	s_add_u32 s38, s12, s8
	s_load_dwordx2 s[36:37], s[36:37], 0x0
	ds_read2_b32 v[12:13], v15 offset1:1
	s_addc_u32 s39, s13, s9
	s_add_u32 s40, s12, s21
	s_load_dwordx2 s[38:39], s[38:39], 0x0
	s_addc_u32 s41, s13, s22
	s_add_i32 s29, s29, 4
	s_load_dwordx2 s[40:41], s[40:41], 0x0
	s_add_u32 s12, s12, s10
	s_addc_u32 s13, s13, s11
	s_add_i32 s30, s30, 16
	s_cmp_eq_u32 s18, s29
	s_waitcnt vmcnt(3)
	v_lshrrev_b32_e32 v18, 4, v20
	v_and_b32_e32 v19, 15, v20
	s_waitcnt vmcnt(2)
	v_lshrrev_b32_e32 v20, 4, v14
	v_and_b32_e32 v21, 15, v14
	ds_read2_b32 v[14:15], v15 offset0:2 offset1:3
	v_cndmask_b32_e64 v18, v18, v19, s0
	v_cndmask_b32_e64 v20, v20, v21, s0
	s_waitcnt vmcnt(1)
	v_lshrrev_b32_e32 v19, 4, v16
	v_and_b32_e32 v16, 15, v16
	v_cvt_f32_ubyte0_e32 v18, v18
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v21, 4, v17
	v_and_b32_e32 v17, 15, v17
	v_cndmask_b32_e64 v16, v19, v16, s0
	v_cvt_f32_ubyte0_e32 v19, v20
	s_waitcnt lgkmcnt(0)
	v_fma_f32 v18, s34, v18, s35
	v_cndmask_b32_e64 v17, v21, v17, s0
	v_cvt_f32_ubyte0_e32 v16, v16
	v_fma_f32 v19, s36, v19, s37
	v_fmac_f32_e32 v4, v12, v18
	v_cvt_f32_ubyte0_e32 v12, v17
	v_fma_f32 v16, s38, v16, s39
	v_fmac_f32_e32 v4, v13, v19
	v_fma_f32 v12, s40, v12, s41
	v_fmac_f32_e32 v4, v14, v16
	v_fmac_f32_e32 v4, v15, v12
	s_cbranch_scc0 .LBB0_31
; %bb.32:                               ; %Flow302
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s12, s18
	s_andn2_b32 vcc_lo, exec_lo, s20
	s_cbranch_vccz .LBB0_34
	s_branch .LBB0_27
.LBB0_33:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b32_e32 v4, 0
	s_andn2_b32 vcc_lo, exec_lo, s20
	s_cbranch_vccnz .LBB0_27
.LBB0_34:                               ; %.epil.preheader.preheader
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_lshl_b32 s13, s12, 2
	s_mul_i32 s30, s7, s12
	s_add_i32 s29, s13, 0
	s_mul_hi_u32 s13, s6, s12
	s_mul_i32 s12, s6, s12
	s_add_i32 s13, s13, s30
	s_add_u32 s12, s4, s12
	s_addc_u32 s13, s5, s13
	s_mov_b32 s30, s16
.LBB0_35:                               ; %.epil.preheader
                                        ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_co_u32 v7, vcc_lo, s12, v0
	v_add_co_ci_u32_e32 v8, vcc_lo, s13, v6, vcc_lo
	s_load_dwordx2 s[34:35], s[12:13], 0x0
	s_waitcnt_vscnt null, 0x0
	global_load_ubyte v5, v[7:8], off
	v_mov_b32_e32 v7, s29
	s_add_i32 s29, s29, 4
	s_add_u32 s12, s12, s6
	s_addc_u32 s13, s13, s7
	s_add_i32 s30, s30, -1
	ds_read_b32 v7, v7
	s_cmp_lg_u32 s30, 0
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v8, 4, v5
	v_and_b32_e32 v5, 15, v5
	v_cndmask_b32_e64 v5, v8, v5, s0
	v_cvt_f32_ubyte0_e32 v5, v5
	s_waitcnt lgkmcnt(0)
	v_fma_f32 v5, s34, v5, s35
	v_fmac_f32_e32 v4, v7, v5
	s_cbranch_scc1 .LBB0_35
	s_branch .LBB0_27
.LBB0_36:                               ; %.loopexit
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attention_hfq4_kv
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 22
		.amdhsa_next_free_sgpr 42
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
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
.Lfunc_end0:
	.size	attention_hfq4_kv, .Lfunc_end0-attention_hfq4_kv
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2448
; NumSgprs: 44
; NumVgprs: 22
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 44
; NumVGPRsForWavesPerEU: 22
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 6, 3214868480
	.fill 48, 4, 3214868480
	.type	__hip_cuid_e20926ac8ddbfbe2,@object ; @__hip_cuid_e20926ac8ddbfbe2
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_e20926ac8ddbfbe2
__hip_cuid_e20926ac8ddbfbe2:
	.byte	0                               ; 0x0
	.size	__hip_cuid_e20926ac8ddbfbe2, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.4 25012 e5bf7e55c91490b07c49d8960fa7983d864936c4)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_e20926ac8ddbfbe2
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         184
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           attention_hfq4_kv
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         attention_hfq4_kv.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     22
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1010
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
