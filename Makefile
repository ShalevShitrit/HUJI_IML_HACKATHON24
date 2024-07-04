# Makefile for creating a tar file with the specified structure

STUDENT_ID = 314825597
TAR_FILE = $(STUDENT_ID).tar

# List of files and directories to include in the tar file
FILES = USERS.txt README.txt project.pdf 1

all: $(TAR_FILE)

$(TAR_FILE): $(FILES)
	@echo "Creating tar file $(TAR_FILE)"
	tar -cvf $(TAR_FILE) $(FILES)

clean:
	@echo "Cleaning up..."
	rm -f $(TAR_FILE)

.PHONY: all clean