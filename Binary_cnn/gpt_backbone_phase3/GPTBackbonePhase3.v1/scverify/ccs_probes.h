#ifndef _INCLUDED_CCS_PROBES_H_
#define _INCLUDED_CCS_PROBES_H_
#include <ccs_probe_comparator.h>

// ac::probe() monitors are added during the extract stage
//  when synthesized user probes are found.
template<int Tw>
inline void ccs_probe_fifo_put(std::string prbnm, sc_lv<Tw> &val)
{
  #ifdef SCVERIFY_VERBOSE_AC_PROBE
  std::cout << "NO SRC PROBE VALUE CAPTURED for '" << prbnm << "'" << std::endl;
  #endif
}

#endif //_INCLUDED_CCS_PROBES_H_


