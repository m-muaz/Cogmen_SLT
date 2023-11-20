# Add a target for deleting the work_dirs/speech_cogmen directory during
# testing
# ------------------------------------------------------------------------------
.PHONY: clean

clean:
	rm -rf work_dirs/speech_cogmen
	trash-empty
# print that the clean target has been completed
	@echo "Clean target completed"
# ------------------------------------------------------------------------------