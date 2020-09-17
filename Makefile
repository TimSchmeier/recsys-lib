SHELL := /bin/bash

REPO_NAME := recsyslib

init:
	pre-commit install

test:
	-pre-commit run --all-files
	tox

install:
	pip install -e .

# RELEASE_TYPE = {patch, minor, major}
bump:
	bump2version $(RELEASE_TYPE)
	git push --tags

# must have a github personal authentication token available in the environment as `GITHUB_PAT`, export it in your bash_profile
release:
	$(eval TAG=$(shell git describe --abbrev=0))
	$(eval PAYLOAD='{"tag_name": "$(TAG)", "target_commitish": "master", "name": "$(TAG)", "body": "Release of version $(TAG)", "draft": false, "prerelease": false}')
	curl -H 'Authorization: token $(GITHUB_PAT)' --data $(PAYLOAD) https://api.github.com/repos/TimSchmeier/$(REPO_NAME)/releases

.PHONY: init test install bump release
