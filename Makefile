CMAKE := cmake
GENERATOR := Ninja
JOBS := $(shell nproc)

BUILD_ROOT := Build

DIR_DEBUG := $(BUILD_ROOT)/Debug
DIR_DEV := $(BUILD_ROOT)/Development

# Lab to run, e.g. make run lab=Lab01
lab ?= Lab01
LAB_ARGS ?=

.DEFAULT_GOAL := build

.PHONY: help configure configure-debug configure-dev \
		makedir-debug makedir-dev \
		build build-debug build-dev \
		run run-debug list format tidy clean

help:
	@echo 'make                          build Development'
	@echo 'make build-debug              build Debug'
	@echo 'make run lab=Lab01            build Development and run Lab01'
	@echo 'make run-debug lab=Lab01      build Debug and run Lab01'
	@echo 'make run lab=Lab01 LAB_ARGS=  pass arguments to the lab'
	@echo 'make list                     list available labs'
	@echo 'make format                   clang-format every tracked source file'
	@echo 'make tidy                     clang-tidy every tracked source file'
	@echo 'make clean                    remove the Build directory'

configure: configure-debug configure-dev

configure-debug: makedir-debug
	@$(CMAKE) -B $(DIR_DEBUG) -G $(GENERATOR) -DCMAKE_BUILD_TYPE=Debug -Wno-deprecated
	@ln -sf $(DIR_DEBUG)/compile_commands.json compile_commands.json

makedir-debug:
	@mkdir -pv $(DIR_DEBUG)

configure-dev: makedir-dev
	@$(CMAKE) -B $(DIR_DEV) -G $(GENERATOR) -DCMAKE_BUILD_TYPE=Development -Wno-deprecated
	@ln -sf $(DIR_DEV)/compile_commands.json compile_commands.json

makedir-dev:
	@mkdir -pv $(DIR_DEV)

build: build-dev

build-debug:
	@$(CMAKE) --build $(DIR_DEBUG) -j $(JOBS)
	@ln -sf $(DIR_DEBUG)/compile_commands.json compile_commands.json

build-dev:
	@$(CMAKE) --build $(DIR_DEV) -j $(JOBS)
	@ln -sf $(DIR_DEV)/compile_commands.json compile_commands.json

run: run-dev

run-debug: build-debug
	@$(MAKE) --no-print-directory _run DIR=$(DIR_DEBUG)

run-dev: build-dev
	@$(MAKE) --no-print-directory _run DIR=$(DIR_DEV)

# Runs Labs/$(lab), the directory and the executable are both named LabNN
.PHONY: _run
_run:
	@if [ ! -d Labs/$(lab) ]; then \
		echo "no lab 'Labs/$(lab)', available labs:"; \
		ls -d Labs/Lab*/ 2>/dev/null | sed 's:/$$::' || echo '  none'; \
		exit 1; \
	fi; \
	echo '> running $(lab)'; \
	./$(DIR)/Target/$(lab) $(LAB_ARGS)

list:
	@ls -d Labs/Lab*/ 2>/dev/null | sed 's:/$$::' || echo 'no labs yet'

format:
	@git ls-files "*.cpp" "*.h" | xargs clang-format -i
	@echo Ok.

tidy:
	@git ls-files "*.cpp" | xargs clang-tidy -p compile_commands.json
	@echo Ok.

clean:
	@rm -rf $(BUILD_ROOT)
