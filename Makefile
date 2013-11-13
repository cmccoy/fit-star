BUILD := build

all:
	mkdir -p $(BUILD) && (cd $(BUILD) && cmake ..)
	+make -C$(BUILD)

clean:
	rm -rf $(BUILD)

style:
	astyle  -A3 \
	        --pad-oper \
	        --unpad-paren \
	        --keep-one-line-blocks \
	        --keep-one-line-statements \
	        --suffix=none \
	        --formatted \
	        --lineend=linux \
					--align-reference=type \
					--align-pointer=type \
					--indent-switches \
	        `find src -regextype posix-extended -regex ".*\.[ch]pp$$"`

.PHONY: all clean style
