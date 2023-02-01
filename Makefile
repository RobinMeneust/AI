# Directories
srcdir = ./src/
bindir = ./bin/
docdir = ./doc/
savedir = ./save/

# Compilation variables
DATE = $(shell date +'%Y_%m_%d_%Hh%M')
AUTHOR = meneust_robin
CC = g++ -Wall
CP = cp -r
RM = rm -rf
PROG = AI
SRC = $(wildcard $(srcdir)*.cpp)
OBJ = $(subst $(srcdir), $(bindir), $(SRC:.cpp=.o))
HEAD = $(wildcard ./include/*.h)
LDLIBS = -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs
DEBUGFLAG = -g


all : $(bindir)$(PROG)

$(bindir)$(PROG) : $(OBJ)
	$(CC) $(OBJ) -o $(bindir)$(PROG) $(LDLIBS) $(DEBUGFLAG)

$(bindir)%.o : $(srcdir)%.cpp $(HEAD)
	$(CC) -c $< -o $@ $(DEBUGFLAG)

# Remove temporary files
clean : cleanConfirm
	$(RM) *\*~ *.bak *.old \#*\# $(bindir)*.o 
	

# Generate the doxygen documentation
doc: createDoxyfile
	doxygen $(docdir)Doxyfile
	
		
.PHONY: cleanConfirm cleanDoc save saveByDate createDoxyfile restore give run


# Ask for confirmation to delete temporary files
cleanConfirm :
	@ echo "Do you want to execute : \"$(RM) $(bindir)*.o *.\*~ *.bak *.old \#*\# \" ? (y/n)" && read answer && [ $${answer} = "y" ]
	
# Remove all files in $(docdir)
cleanDoc :
	$(RM) $(docdir)*

# Create a backup for this project
save:
	$(CP) $(srcdir). $(savedir)backup/

# Create a backup for this project with the current date in its name
saveByDate:
	mkdir "$(savedir)/backup_$(DATE)/"
	cp -r $(srcdir). "$(savedir)/backup_$(DATE)/"
	
createDoxyfile:
	if [ ! -d $(docdir) ]; then mkdir $(docdir); fi; if [ ! -f $(docdir)Doxyfile ]; then doxygen -g $(docdir)Doxyfile && sed -i 's@^OUTPUT_DIRECTORY.*=$$@OUTPUT_DIRECTORY = $(docdir)@' $(docdir)Doxyfile && sed -i 's@^RECURSIVE.*= NO$$@RECURSIVE = YES@' $(docdir)Doxyfile; fi;

# Restore the backup in save/backup
restore:
	cp -ir $(savedir)backup/. $(srcdir)

# Generate the archive containing this project
give:
	mkdir "meneust_robin_$(PROG)"
	mkdir "meneust_robin_$(PROG)/bin" "meneust_robin_$(PROG)/save"
	cp -rt "meneust_robin_$(PROG)" ./Makefile $(docdir) $(srcdir)
	zip -r meneust_robin_$(PROG).zip meneust_robin_$(PROG)
	rm -r "meneust_robin_$(PROG)"

# Run the executable without any arguments
run:
	${bindir}${PROG}