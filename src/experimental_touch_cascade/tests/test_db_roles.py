from experimental_touch_cascade.db import connect, init_evidence_db, init_run_db, require_role
import pytest


def test_db_roles_are_not_interchangeable(tmp_path):
    e=tmp_path/'e.sqlite'; r=tmp_path/'r.sqlite'
    init_evidence_db(e); init_run_db(r,'p','fp')
    ce=connect(e); cr=connect(r)
    require_role(ce,'public_evidence'); require_role(cr,'candidate_run_state')
    with pytest.raises(RuntimeError): require_role(ce,'candidate_run_state')
    with pytest.raises(RuntimeError): require_role(cr,'public_evidence')
    ce.close(); cr.close()
