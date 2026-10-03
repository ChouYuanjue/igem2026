from pathlib import Path

from projects.active.fibre.evidence.enrichment import (
    AdapterResult, DiscoveredResource, EnrichmentSeed, EnrichmentStore, run_discovery,
)


class _ChainAdapter:
    name='chain'
    accepted_kinds=frozenset({'root','child'})

    def discover(self,seed,*,limit=10,materialize=False):
        if seed.kind=='root':
            resource=DiscoveredResource(
                source='chain',kind='node',canonical_key='root-resource',
                payload={'value':seed.value},
            )
            child=EnrichmentSeed(
                kind='child',value='C1',origin='chain',
                parent_resource_id=resource.resource_id,
            )
            return AdapterResult((resource,),(child,))
        resource=DiscoveredResource(
            source='chain',kind='node',canonical_key='child-resource',
            payload={'value':seed.value},
        )
        return AdapterResult((resource,),())


class _DuplicateAdapter:
    name='duplicate'
    accepted_kinds=frozenset({'root'})

    def discover(self,seed,*,limit=10,materialize=False):
        resource=DiscoveredResource(
            source='same',kind='node',canonical_key='X',payload={'x':1}
        )
        return AdapterResult((resource,),())


def test_discovery_persists_child_edges_and_resumes_bfs(tmp_path: Path):
    root=EnrichmentSeed(kind='root',value='R')
    store=EnrichmentStore(tmp_path/'graph')
    try:
        first=run_discovery(
            [root],[_ChainAdapter()],store,max_depth=0,max_resources=10
        )
        assert first['resource_count']==1
        assert first['seed_edge_count']==1
    finally:
        store.close()

    # New process/store instance: the root adapter run is cached, but its child
    # edge must still be replayed so a deeper invocation can continue outward.
    store=EnrichmentStore(tmp_path/'graph')
    try:
        second=run_discovery(
            [root],[_ChainAdapter()],store,max_depth=1,max_resources=10
        )
        assert second['resource_count']==2
        assert second['seed_count']==2
    finally:
        store.close()


def test_resource_budget_counts_unique_resources_not_duplicate_discovery_events(tmp_path: Path):
    root=EnrichmentSeed(kind='root',value='R')
    store=EnrichmentStore(tmp_path/'graph')
    try:
        summary=run_discovery(
            [root],[_DuplicateAdapter(),_DuplicateAdapter()],store,
            max_depth=0,max_resources=10,refresh=True,
        )
        assert summary['resource_count']==1
        assert summary['new_resource_events']==1
    finally:
        store.close()


class _MaterializingAdapter:
    name='materializing'
    accepted_kinds=frozenset({'root'})

    def discover(self,seed,*,limit=10,materialize=False):
        rows=[DiscoveredResource(
            source='materializing',kind='metadata',canonical_key='M',payload={}
        )]
        if materialize:
            rows.append(DiscoveredResource(
                source='materializing',kind='file',canonical_key='M:file',payload={},
                blob_text='coordinate data',blob_suffix='.cif',
            ))
        return AdapterResult(tuple(rows),())


class _ThreeResourceAdapter:
    name='three'
    accepted_kinds=frozenset({'root'})

    def discover(self,seed,*,limit=10,materialize=False):
        return AdapterResult(tuple(
            DiscoveredResource(source='three',kind='node',canonical_key=str(i),payload={'i':i})
            for i in range(3)
        ),())


def test_metadata_cache_does_not_block_later_materialization(tmp_path: Path):
    root=EnrichmentSeed(kind='root',value='R')
    store=EnrichmentStore(tmp_path/'graph')
    try:
        run_discovery([root],[_MaterializingAdapter()],store,max_depth=0,max_resources=10)
        assert store.summary()['resource_count']==1
        run_discovery(
            [root],[_MaterializingAdapter()],store,max_depth=0,max_resources=10,
            materialize=True,
        )
        assert store.summary()['resource_count']==2
        assert store.run_status(root,'materializing')=='ok'
        assert store.run_status(root,'materializing:materialize')=='ok'
    finally:
        store.close()


def test_budget_truncated_adapter_batch_resumes_on_next_run(tmp_path: Path):
    root=EnrichmentSeed(kind='root',value='R')
    store=EnrichmentStore(tmp_path/'graph')
    try:
        first=run_discovery([root],[_ThreeResourceAdapter()],store,max_depth=0,max_resources=1)
        assert first['resource_count']==1
        assert store.run_status(root,'three')=='partial_budget'
        second=run_discovery([root],[_ThreeResourceAdapter()],store,max_depth=0,max_resources=10)
        assert second['resource_count']==3
        assert store.run_status(root,'three')=='ok'
    finally:
        store.close()


def test_seed_normalization_deduplicates_case_and_doi_forms(tmp_path: Path):
    a=EnrichmentSeed(kind='doi',value='https://doi.org/10.1000/ABC.1')
    b=EnrichmentSeed(kind='doi',value='10.1000/abc.1')
    assert a.seed_id==b.seed_id
    u1=EnrichmentSeed(kind='uniprot',value='q93yv0')
    u2=EnrichmentSeed(kind='uniprot',value='Q93YV0')
    assert u1.seed_id==u2.seed_id
