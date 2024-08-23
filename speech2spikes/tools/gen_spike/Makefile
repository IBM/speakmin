
all: spikes plots analysis

spikes :
	make -f speech2spikes.mk

plots :
	make -f plot_spikes.mk

analysis :
	make -f analysis.mk

archive :
	make -f speech2spikes.mk archive
	make -f plot_spikes.mk archive
	make -f analysis.mk archive

show_category:
	@echo "zero one two three four five six seven eight nine yes no up down left right on off stop go"
	@echo "cat dog bird house bed tree happy wow sheila marvin forward backward follow learn visual _background_noise_"

clean :
	-@make -f speech2spikes.mk clean
	-@make -f plot_spikes.mk clean
	-@make -f analysis.mk clean

.PHONY : spikes plots analysis archive all
