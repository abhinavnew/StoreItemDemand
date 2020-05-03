library(shiny)

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
                       titlePanel("Store Item Forecasting problem-File Upload Page"),
  
  # Sidebar with a slider input for number of bins 
           sidebarLayout(
           sidebarPanel(
      
            h3("This is the side panel"),
            
    # # Input: Select a file ----
               fileInput("file1", "Choose CSV File",
              multiple = FALSE,
              accept = c("text/csv",
                         "text/comma-separated-values,text/plain",
                         ".csv")),
   
     ##horizontal line
       tags$hr(),
    # # ##Checkbox for checking if file has a header
    checkboxInput("header", "Header", TRUE),
    ##radio button for separator
    radioButtons(
      "sep",
      "Separator",
      choices = c(
        Comma = ",",
        Semicolon = ";",
        Tab = "\t"
      ),
      selected = ","
                 ),
    
   ## Input: Select quotes ----
    radioButtons(
      "quote",
      "Quote",
      choices = c(
        None = "",
        "Double Quote" = '"',
        "Single Quote" = "'"
      ),
      selected = '"'
                ),
    
   ##Horizontal line
   
     tags$hr(),
   
    #  # Input: Select number of rows to display ----
    radioButtons("disp",
                 "Display",
                 choices = c(Head = "head",
                             All = "all"),
                 selected = "head")

               ),
   ##closed sidebar panel
   
    # Show a plot of the generated distribution
    mainPanel(
      h3("Contents of the Input File are :"),
       ##Output :data file 
      tableOutput("contents")
              
      )
           )
  ##closed sidebar layout
  
  ##closing fluid and shinyui func
))
