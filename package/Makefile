ARCH ?= amd64
REV := $(shell git describe --always)

upload: package
	bundle exec deb-s3 upload --sign 3AA09EC1 -a amd64  *.deb --bucket cmccoy-debian-repo

#FITSTAR_VERSION := $(shell git describe)
FITSTAR_VERSION := 0.1.0

package: fit-star_$(FITSTAR_VERSION)_$(ARCH).deb

fit-star_$(FITSTAR_VERSION)_$(ARCH).deb:
	@echo "Packaging $@"
	rm -rf /tmp/fit-star-install
	mkdir -p /tmp/fit-star-install
	mkdir -p deb-build
	(cd deb-build && cmake ../.. -DCMAKE_INSTALL_PREFIX=/usr && make && make install DESTDIR=/tmp/fit-star-install)
	bundle exec fpm \
		-d libhmsbeagle1 \
		-d libgomp1 \
		-d "libstdc++6 (>= 4.4.0)" \
		-d "libbpp-core2 (>= 2.1.0)" \
		-d "libbpp-seq9 (>= 2.1.0)" \
		-d "libbpp-phyl9 (>= 2.1.0)" \
		-t deb \
		-s dir \
		-n fit-star \
		-v $(FITSTAR_VERSION) \
		-C /tmp/fit-star-install \
		--license GPLv3 \
		--maintainer "Connor McCoy <cmccoy@fhcrc.org>" \
		--description "Fit star-tree models. (build of $(REV))" \
		--url "http://github.com/cmccoy/fit-star" \
		--category universe/math \
		usr/bin

.PHONY: package upload
