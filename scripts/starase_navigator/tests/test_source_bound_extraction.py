from __future__ import annotations

import json
from unittest.mock import patch

from scripts.starase_navigator.routing.language import DeepSeekResolver


class FakeResponse:
    def __init__(self, payload):
        self.payload=payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_source_bound_extraction_rejects_non_source_spans_and_unbound_scope():
    resolver=DeepSeekResolver()
    captured={}
    response={
        'id':'source-bound-test',
        'choices':[{'message':{'content':json.dumps({
            'paragraph_role':'catalytic_assay',
            'facts':[
                {
                    'type':'pH',
                    'evidence_text':'50 mM HEPES, pH 7.2',
                    'scope_text':'enzyme activity assay',
                    'scope_resolved':True,
                    'target_ids':['T1','NOT_ALLOWED'],
                },
                {
                    'type':'temperature',
                    'evidence_text':'incubated at 99°C',
                    'scope_text':'',
                    'scope_resolved':False,
                },
                {
                    'type':'buffer',
                    'evidence_text':'50 mM HEPES',
                    'scope_text':'not in source',
                    'scope_resolved':True,
                },
            ]
        })}}],
    }

    def fake_post(url,**kwargs):
        captured['url']=url
        captured['json']=kwargs['json']
        captured['timeout']=kwargs['timeout']
        return FakeResponse(response)

    resolver.session.post=fake_post
    source='The enzyme activity assay used 50 mM HEPES, pH 7.2 and was incubated at 30°C.'
    with patch.dict('os.environ',{'DEEPSEEK_API_KEY':'test-key','DEEPSEEK_MODEL':'deepseek-flash'}):
        result=resolver.extract_source_bound_facts(
            source,
            source_context={'section_title':'Enzyme assay'},
            target_context=[{
                'target_id':'T1',
                'enzyme_name':'TPS1',
                'substrate_name':'GPP',
                'product_name':'product',
            }],
        )

    assert result['status']=='ok'
    assert result['paragraph_role']=='catalytic_assay'
    assert result['raw_fact_count']==3
    assert result['rejected_fact_count']==1
    assert len(result['facts'])==2
    assert result['facts'][0]=={
        'type':'pH',
        'evidence_text':'50 mM HEPES, pH 7.2',
        'scope_text':'enzyme activity assay',
        'scope_resolved':True,
        'target_ids':[],
        'target_assignment_status':'unresolved',
        'source_span_verified':True,
    }
    assert result['facts'][1]['type']=='buffer'
    assert result['facts'][1]['scope_text']==''
    assert result['facts'][1]['scope_resolved'] is False
    assert result['facts'][1]['target_ids']==[]
    assert result['facts'][1]['target_assignment_status']=='unresolved'
    assert captured['json']['thinking']=={'type':'disabled'}
    assert captured['json']['stream'] is False
    assert captured['timeout']==60


def test_source_bound_extraction_binds_target_only_when_source_names_it():
    resolver=DeepSeekResolver()
    response={
        'id':'source-bound-explicit-target-test',
        'choices':[{'message':{'content':json.dumps({
            'paragraph_role':'catalytic_assay',
            'facts':[{
                'type':'temperature',
                'evidence_text':'TPS1 was assayed at 30°C',
                'scope_text':'TPS1',
                'scope_resolved':True,
                'target_ids':['T1'],
            }],
        })}}],
    }
    resolver.session.post=lambda *_a,**_k: FakeResponse(response)
    source='TPS1 was assayed at 30°C with GPP as substrate.'
    with patch.dict('os.environ',{'DEEPSEEK_API_KEY':'test-key'}):
        result=resolver.extract_source_bound_facts(
            source,
            target_context=[{
                'target_id':'T1',
                'enzyme_name':'TPS1',
                'substrate_name':'GPP',
            }],
        )
    assert result['facts'][0]['target_ids']==['T1']
    assert result['facts'][0]['target_assignment_status']=='candidate'


def test_source_bound_extraction_discards_non_catalytic_workflow_facts():
    resolver=DeepSeekResolver()
    response={
        'id':'non-catalytic-test',
        'choices':[{'message':{'content':json.dumps({
            'paragraph_role':'non_catalytic_workflow',
            'facts':[{
                'type':'temperature',
                'evidence_text':'grown at 37°C',
                'scope_text':'expression culture',
                'scope_resolved':True,
                'target_ids':['T1'],
            }],
        })}}],
    }
    resolver.session.post=lambda *_a,**_k: FakeResponse(response)
    with patch.dict('os.environ',{'DEEPSEEK_API_KEY':'test-key'}):
        result=resolver.extract_source_bound_facts(
            'The recombinant culture was grown at 37°C before purification.',
            source_context={'section_title':'Protein expression'},
            target_context=[{'target_id':'T1','enzyme_name':'TPS1'}],
        )
    assert result['status']=='ok'
    assert result['paragraph_role']=='non_catalytic_workflow'
    assert result['raw_fact_count']==1
    assert result['rejected_fact_count']==1
    assert result['facts']==[]


def test_source_bound_extraction_requires_configuration_without_network_call():
    resolver=DeepSeekResolver()
    resolver.session.post=lambda *_a,**_k: (_ for _ in ()).throw(AssertionError('network should not run'))
    with patch.dict('os.environ',{},clear=True):
        result=resolver.extract_source_bound_facts('pH 7.0')
    assert result=={'facts':[],'status':'not_configured'}
