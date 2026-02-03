.PHONY: update

# Update local repository to latest origin/dev
update:
	git fetch origin
	git checkout dev
	git pull origin dev
