"""Saved catalogs and profiles are reused only for the same key, endpoint and age."""

import time

from slipagent import startupcache
from slipagent.capabilities import ModelCapabilities, RequestProfile
from slipagent.types import ModelInfo


def test_catalogs_require_every_provider_with_a_matching_fresh_identity(monkeypatch):
    models = [ModelInfo("a", context_length=10, pricing={"prompt": "0"}), ModelInfo("b", provider="nvidia")]
    startupcache.save_catalogs(models, {"openrouter": "key-1", "nvidia": "key-2"})
    restored = startupcache.cached_catalog({"openrouter": "key-1", "nvidia": "key-2"})
    assert [(model.provider, model.id, model.context_length) for model in restored] == [
        ("openrouter", "a", 10), ("nvidia", "b", None)]
    assert startupcache.cached_catalog({"openrouter": "other-key"}) is None
    assert startupcache.cached_catalog({"openrouter": "key-1", "google": "key-3"}) is None
    later = time.time() + startupcache.MAX_AGE_SECONDS + 1
    monkeypatch.setattr(startupcache.time, "time", lambda: later)
    assert startupcache.cached_catalog({"openrouter": "key-1"}) is None


def test_profiles_round_trip_including_endpoint_routing():
    model = ModelInfo("router/model")
    profile = ModelCapabilities({}, [{"tag": "stub", "supported_parameters": ["tools", "max_tokens"],
                                      "context_length": 4096}])
    startupcache.save_profile(model, "identity", profile)
    restored = startupcache.cached_profile(model, "identity")
    assert type(restored) is ModelCapabilities and vars(restored) == vars(profile)
    assert restored.provider_preferences() == profile.provider_preferences()
    assert startupcache.cached_profile(model, "other") is None
    plain = RequestProfile(parameters={"tools"}, context_length=10)
    startupcache.save_profile(ModelInfo("x", provider="google"), "id", plain)
    assert vars(startupcache.cached_profile(ModelInfo("x", provider="google"), "id")) == vars(plain)


def test_corrupt_cache_is_ignored():
    startupcache.cache_path().parent.mkdir(parents=True, exist_ok=True)
    startupcache.cache_path().write_text("{not json")
    assert startupcache.cached_catalog({"openrouter": "key"}) is None
    startupcache.save_catalogs([ModelInfo("a")], {"openrouter": "key"})
    assert startupcache.cached_catalog({"openrouter": "key"})[0].id == "a"


def test_profiles_from_another_build_are_ignored(monkeypatch):
    from slipagent import startupcache
    from slipagent.capabilities import RequestProfile
    from slipagent.types import ModelInfo
    model = ModelInfo("vendor/model", provider="openrouter")
    startupcache.save_profile(model, "identity", RequestProfile(parameters={"tools"}, context_length=1000))
    assert startupcache.cached_profile(model, "identity") is not None
    monkeypatch.setattr(startupcache, "__version__", "0.0.0", raising=False)
    import slipagent
    monkeypatch.setattr(slipagent, "__version__", "0.0.0")
    assert startupcache.cached_profile(model, "identity") is None
