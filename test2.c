AMGX_RC AMGX_vector_create_impl(AMGX_vector_handle *vec, AMGX_resources_handle rsc, AMGX_Mode mode)
    {
        nvtxRange nvrf(__func__);

        AMGX_CPU_PROFILER( "AMGX_vector_create " );
        Resources *resources = NULL;
        AMGX_ERROR rc = AMGX_OK;
        AMGX_ERROR rc_vec = AMGX_OK;

        AMGX_TRIES()
        {
            ResourceW c_r(rsc);

            if (!c_r.wrapped()) { AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_PARAMETERS, NULL); }

            resources = c_r.wrapped().get();
            cudaSetDevice(resources->getDevice(0));

            switch (mode)
            {
#define AMGX_CASE_LINE(CASE) case CASE: { \
      auto* wvec = create_managed_mode_object<CASE,Vector,AMGX_vector_handle>(vec, mode); \
      rc_vec = wvec->is_valid() ? AMGX_OK : AMGX_ERR_UNKNOWN; \
      wvec->wrapped()->setResources(resources); \
      } \
      break;
                    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
                    AMGX_FORINTVEC_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

                default:
                    AMGX_CHECK_API_ERROR(AMGX_ERR_BAD_MODE, resources)
            }
        }

        AMGX_CATCHES(rc)
        AMGX_CHECK_API_ERROR(rc_vec, resources)
        return getCAPIerror_x(rc);
    }


